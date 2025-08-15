import asyncio
import asyncpg
import argparse
from typing import Optional

from bat_util import (
    DB_CONFIG,
    ai_identify_transmission,
    log_health,
    get_gemini_rpm,
    suggest_ai_concurrency,
    perf,
    async_timed,
    print_perf_summary,
)


async def fetch_candidates(
    pool: asyncpg.Pool,
    source: Optional[str],
    limit: Optional[int],
) -> list[asyncpg.Record]:
    where = "manual IS NULL"
    params = []
    if source:
        where += " AND source = $1"
        params.append(source)

    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    sql = f"""
        SELECT id, title, excerpt
        FROM auction
        WHERE {where}
        ORDER BY id DESC
        {limit_clause}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params) if params else await conn.fetch(sql)
    return rows


async def process_row(
    pool: asyncpg.Pool,
    row: asyncpg.Record,
    dry_run: bool = False,
) -> int:
    """Returns 1 if updated, 0 otherwise."""
    title = (row["title"] or "").strip()
    excerpt = (row["excerpt"] or "").strip()

    try:
        with perf("backfill.ai_call"):
            auto_flag = await ai_identify_transmission(title, excerpt)
    except Exception as e:
        print(f"AI error for id={row['id']}: {e}")
        return 0

    if auto_flag is None:
        return 0

    manual = not auto_flag  # True for manual cars in DB
    if dry_run:
        print(f"DRY-RUN: would set manual={manual} for id={row['id']}")
        return 1

    with perf("backfill.db_update"):
        async with pool.acquire() as conn:
            await conn.execute("UPDATE auction SET manual=$1 WHERE id=$2", manual, row["id"])
    return 1


async def backfill_manual(
    source: Optional[str],
    limit: Optional[int],
    concurrency: int,
    dry_run: bool,
) -> None:
    # Size the pool relative to concurrency to avoid exhausting connections.
    max_pool = max(4, min(32, concurrency // 2))
    pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=max_pool)
    try:
        with perf("backfill.db_fetch_candidates"):
            rows = await fetch_candidates(pool, source, limit)
        if not rows:
            print("Backfill: no rows with manual IS NULL (matching filters).")
            return

        sem = asyncio.Semaphore(concurrency)

        async def worker(r: asyncpg.Record) -> int:
            async with sem:
                return await process_row(pool, r, dry_run=dry_run)

        updated = 0
        CHUNK = max(100, concurrency * 2)
        for i in range(0, len(rows), CHUNK):
            batch = rows[i : i + CHUNK]
            with perf("backfill.batch_gather"):
                results = await asyncio.gather(*(worker(r) for r in batch), return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    print(f"Backfill exception: {res}")
                else:
                    updated += int(res)
            print(f"Processed {min(i+CHUNK, len(rows))}/{len(rows)} rows...")

        print(f"Backfill complete: updated {updated}/{len(rows)} rows")
    finally:
        await pool.close()


async def main():
    parser = argparse.ArgumentParser(description="Backfill auction.manual using AI.")
    parser.add_argument("--source", help="Filter by source (e.g., 'bat')", default=None)
    parser.add_argument("--limit", type=int, help="Max rows to process", default=None)
    parser.add_argument("--concurrency", type=int, help="Concurrent AI calls (default auto)", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--log-level", default=None, help="Logging level (e.g., INFO, DEBUG)")

    args = parser.parse_args()

    if args.concurrency is None:
        # Pick a smart default from RPM and a conservative latency.
        # With RPM=4000, this will typically choose around 32–48.
        args.concurrency = suggest_ai_concurrency()
        print(f"Auto concurrency set to {args.concurrency} based on GEMINI_RATE_LIMIT_RPM={get_gemini_rpm()}")

    log_health()

    await backfill_manual(
        source=args.source,
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    print_perf_summary()


if __name__ == "__main__":
    asyncio.run(main())
