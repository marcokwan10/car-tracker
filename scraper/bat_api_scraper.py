# bat_api_scraper.py
import asyncio
import aiohttp
import re
from datetime import datetime

from bat_util import (
    _ai_fallback_count,
    split_make_model,
    extract_mileage,
    extract_mileage_from_detail_page,
    extract_price_and_status,
    parse_sold_date,
    save_to_db,
    log_health,
)


async def fetch_json_page(session: aiohttp.ClientSession, url: str, page: int, per_page: int = 60):
    params = {
        "page": page,
        "per_page": per_page,
        "get_items": 1,
        "get_stats": 0,
        "sort": "td",
    }
    headers = {"Accept": "application/json"}
    async with session.get(url, params=params, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.json()


async def parse_json_listing(item, session: aiohttp.ClientSession):
    # 1. Grab the title & excerpt
    title = item.get("title", "").strip()
    excerpt = item.get("excerpt", "").strip()
    url = item.get("permalink") or item.get("url")

    # 2. Year
    match = re.search(r"(?:19|20)\d{2}", title)
    year = int(match.group()) if match else None
    if not year:
        return None  # Skip listing with no year

    # 3. Make & model (Hybrid make/model extraction, fallback to use Gemini)
    make, model = await split_make_model(title)
    if not make or not model:
        return None  # Skip non-vehicle listings like wheels, hardtops, etc

    # 4. Mileage
    #    Mileage priority: title > excerpt > detail page
    mileage = extract_mileage(title) or extract_mileage(excerpt)
    if mileage is None and url:
        mileage = await extract_mileage_from_detail_page(session, url)

    # 5. Price & status from sold_text (strip any HTML) e.g. "Sold for USD $19,200 on 8/7/25"
    sold_html = item.get("sold_text") or ""
    sold_plain = re.sub(r"<[^>]+>", "", sold_html)
    price, status = extract_price_and_status(sold_plain)

    # 6. Sold date: prefer date embedded in sold_text; fallback to timestamp_end
    sold_date = None
    m = re.search(r"on\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", sold_plain)
    if m:
        sold_date = parse_sold_date(m.group(1))
    if not sold_date:
        ts = item.get("timestamp_end")
        if isinstance(ts, (int, float)):
            sold_date = datetime.fromtimestamp(ts).date()

    return {
        "source": "bat",
        "source_listing_id": str(item.get("id")) if item.get("id") is not None else None,
        "url": url,
        "title": title,
        "year": year,
        "make": make,
        "model": model,
        "original_owner": "original-owner" in title.lower() or "one-owner" in title.lower(),
        "mileage": mileage,
        "sold_price": price,
        "sold_date": sold_date,
        "status": status,
        "excerpt": excerpt,
    }


async def main():
    # Log startup health once
    log_health()

    API_URL = "https://bringatrailer.com/wp-json/bringatrailer/1.0/data/listings-filter"

    all_records = []
    pageLimit = 200
    currentPage = 166
    async with aiohttp.ClientSession() as session:
        while currentPage < pageLimit:
            data = await fetch_json_page(session, API_URL, currentPage)
            listings = data.get("items", [])

            if not listings:
                print(f"No listings on page {currentPage}; stopping.")
                break

            # Concurrently parse each JSON item using the same session
            tasks = [parse_json_listing(item, session) for item in listings]
            parsed_list = await asyncio.gather(*tasks, return_exceptions=True)
            for result in parsed_list:
                if isinstance(result, Exception):
                    print(f"parse_json_listing error: {result}")
                    continue
                if result:
                    all_records.append(result)

            print(f"Page {currentPage}: fetched {len(listings)} items, {len(all_records)} total parsed")
            currentPage += 1

    # Log how many times AI fallback was used
    print(f"AI fallback used {_ai_fallback_count} times")

    if all_records:
        await save_to_db(all_records)
        print(f"Saved {len(all_records)} records to the database.")


if __name__ == "__main__":
    asyncio.run(main())
