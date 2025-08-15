# bat_api_scraper.py
import asyncio
import aiohttp
import re
from datetime import datetime

from bat_util import (
    get_ai_fallback_count,
    reset_ai_fallback_count,
    split_make_model,
    extract_mileage,
    extract_mileage_from_detail_page,
    extract_price_and_status,
    parse_sold_date,
    save_to_db,
    log_health,
    ai_identify_transmission,
)


async def fetch_json_page(
    session: aiohttp.ClientSession,
    url: str,
    page: int,
    per_page: int,
    year: int,
):
    params = {
        "page": page,
        "per_page": per_page,
        "get_items": 1,
        "get_stats": 0,
        "sort": "td",
        "minimum_year": year,
        "maximum_year": year,
    }
    headers = {"Accept": "application/json"}
    async with session.get(url, params=params, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.json()


async def parse_json_listing(item, session: aiohttp.ClientSession, filter_year: int | None = None):
    # 1. Grab the title & excerpt
    title = item.get("title", "").strip()
    excerpt = item.get("excerpt", "").strip()
    url = item.get("permalink") or item.get("url")

    # 2. Year
    match = re.search(r"(?:19|20)\d{2}", title)
    year = int(match.group()) if match else (filter_year if filter_year is not None else None)
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

    # 7. Transmission (AI). ai_identify_transmission returns True for AUTOMATIC, False for MANUAL
    manual = None
    try:
        auto_flag = await ai_identify_transmission(title, excerpt)
        if auto_flag is not None:
            manual = not auto_flag  # store True for manual cars
    except Exception:
        manual = None

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
        "manual": manual,
    }


# Helper to parse a list of listings with error handling
async def parse_listings(listings, session, filter_year):
    tasks = [parse_json_listing(item, session, filter_year=filter_year) for item in listings]
    parsed_list = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for result in parsed_list:
        if isinstance(result, Exception):
            print(f"parse_json_listing error: {result}")
            continue
        if result:
            results.append(result)
    return results


async def main():
    # Log startup health once
    log_health()

    API_URL = "https://bringatrailer.com/wp-json/bringatrailer/1.0/data/listings-filter"

    year = 1950
    result_per_page = 60
    total_ai_fallbacks = 0

    while year < 1980:
        all_records = []
        async with aiohttp.ClientSession() as session:
            # Fetch page 1 to learn total pages for this year
            data_p1 = await fetch_json_page(session, API_URL, page=1, per_page=result_per_page, year=year)
            pages_total = data_p1.get("pages_total") or 1
            items_total = data_p1.get("items_total") or 0
            print(f"[{year}] pages_total={pages_total} items_total={items_total}")

            # Iterate over all pages, reusing the first page response
            for page_num in range(1, int(pages_total) + 1):
                if page_num == 1:
                    data = data_p1
                else:
                    await asyncio.sleep(0.1)  # small politeness delay
                    data = await fetch_json_page(session, API_URL, page=page_num, per_page=result_per_page, year=year)

                listings = data.get("items", [])
                if not listings and page_num > 1:
                    print(f"[{year}] No listings on page {page_num}; stopping early.")
                    break

                parsed = await parse_listings(listings, session, filter_year=year)
                all_records.extend(parsed)
                print(f"[{year}] Page {page_num}: fetched {len(listings)} items, {len(all_records)} total parsed")

        # Log how many times AI fallback was used (live count)
        print(f"AI fallback used {get_ai_fallback_count()} times")
        total_ai_fallbacks += get_ai_fallback_count()
        reset_ai_fallback_count()

        if all_records:
            await save_to_db(all_records)
            print(f"Saved {len(all_records)} records to the database.")

        year += 1

    print(f"Total AI fallbacks across all years: {total_ai_fallbacks}")


if __name__ == "__main__":
    asyncio.run(main())
