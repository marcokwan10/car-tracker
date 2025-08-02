import asyncio
import aiohttp
import asyncpg
import re
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()  # Load .env file

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
}


def extract_price_and_status(text):
    match = re.search(r"(Sold for|Bid to) USD \$([\d,]+)", text)
    if match:
        status = "sold" if match.group(1) == "Sold for" else "reserve_not_met"
        price = int(match.group(2).replace(",", ""))
        return price, status
    return None, None


def parse_sold_date(date_str):
    if not date_str:
        return None
    for fmt in ["%m/%d/%y", "%b %d, %Y"]:  # handle "8/1/25" and "Aug 1, 2025"
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def parse_mileage_string(raw: str) -> int | None:
    """
    Converts a raw mileage string (like '19K', '19,500', '19.5k', '19k-Mile') to an integer mileage.
    Returns None if it cannot be parsed.
    """
    if not raw:
        return None

    # Normalize to lowercase, remove spaces and commas
    s = raw.lower().replace(",", "").strip()
    s = s.replace("-mile", "").replace("mile", "")  # remove '-mile' or 'mile'

    if s.endswith("k"):
        try:
            return int(float(s[:-1]) * 1000)
        except ValueError:
            return None
    try:
        return int(float(s))
    except ValueError:
        return None


def extract_mileage(text: str) -> int | None:
    """
    Extracts the highest plausible odometer reading from string.
    Only matches values with a 'k' suffix or those explicitly followed by 'mile'/'miles'.
    """
    if not text:
        return None

    # Normalize text to lowercase for consistent matching
    text_norm = text.lower()

    # Match either:
    # - numbers with 'k' (e.g. '15k')
    # - numbers followed by 'mile' or 'miles' (e.g. '200 miles', '2,900-Mile')
    pattern = re.compile(r"\b(\d+(?:\.\d+)?k)\b|\b(\d{1,3}(?:[.,]\d{3})*)(?=\s*miles?\b)", re.IGNORECASE)
    matches = pattern.findall(text_norm)

    # Flatten capture groups into raw strings
    candidates = [m[0] or m[1] for m in matches]

    mile_values = []
    for raw in candidates:
        miles = parse_mileage_string(raw)
        if miles is not None:
            mile_values.append(miles)

    return max(mile_values) if mile_values else None


async def extract_mileage_from_detail_page(session: aiohttp.ClientSession, url: str):
    """
    Extract mileage from the 'Listing Details' section of a vehicle page.
    """
    async with session.get(url) as response:
        html = await response.text()

    soup = BeautifulSoup(html, "html.parser")

    # Find <div class="item"> that has a <strong>Listing Details</strong> in it
    detail_sections = soup.find_all("div", class_="item")
    listing_section = None
    for section in detail_sections:
        strong = section.find("strong")
        if strong and "Listing Details" in strong.text:
            listing_section = section
            break

    if not listing_section:
        return None

    # Look for <ul><li> items after that
    ul = listing_section.find("ul")
    if not ul:
        return None

    # Only parse items explicitly mentioning mileage
    for li in ul.find_all("li"):
        text = li.get_text(strip=True)
        # Only parse items explicitly mentioning mileage
        if "mile" not in text.lower():
            continue
        mileage = extract_mileage(text)
        if mileage is not None:
            return mileage

    return None


async def parse_vehicle_listing(listing, session: aiohttp.ClientSession):
    title_el = listing.select_one("h3")
    title = title_el.text.strip() if title_el else ""
    excerpt_el = listing.select_one(".item-excerpt")
    excerpt = excerpt_el.text.strip() if excerpt_el else ""
    url = listing.get("href")

    vehicle_title_match = re.match(
        r"(?:[\d.,kK\-]+-Mile\s+)?(?:Original-Owner,?\s+)?(?P<year>19\d{2}|20\d{2})\s+(?P<make>[A-Z][a-zA-Z]+)\s+(?P<model>[A-Z0-9][\w\-]+(?:\s+\w+)*)",
        title,
    )
    if not vehicle_title_match:
        return None  # skip non-car listings

    year = int(vehicle_title_match.group("year"))
    make = vehicle_title_match.group("make")
    model = vehicle_title_match.group("model")
    original_owner = "original-owner" in title.lower()

    # Mileage priority: title > excerpt > detail page
    mileage = extract_mileage(title) or extract_mileage(excerpt)
    if mileage is None and url:
        mileage = await extract_mileage_from_detail_page(session, url)

    result_el = listing.select_one(".item-results")
    result_text = result_el.text.strip() if result_el else ""
    result_date_str = (
        result_el.select_one("span").text.strip().replace("on ", "")
        if result_el and result_el.select_one("span")
        else None
    )
    result_date = parse_sold_date(result_date_str)

    price, status = extract_price_and_status(result_text)

    return {
        "url": url,
        "title": title,
        "year": year,
        "make": make,
        "model": model,
        "original_owner": original_owner,
        "mileage": mileage,
        "sold_price": price,
        "sold_date": result_date,
        "status": status,
        "excerpt": excerpt,
    }


async def scrape_bring_a_trailer(url: str, max_clicks: int = 100):
    """
    Scrapes Bring a Trailer listings by repeatedly clicking the "Show More" button.
    Uses a more efficient by waiting for new listings to appear rather than a fixed timeout.
    """
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)

        click_count = 0

        while click_count < max_clicks:
            # Count current listings
            current_listings = await page.query_selector_all("a.listing-card")
            current_count = len(current_listings)

            # Find the completed auctions 'Show More' button
            show_more_selectors = [
                ".auctions-container.column-limited-width .auctions-footer-button",  # for all auction result
                ".auctions-completed.page-section .button-show-more",  # for individual model auction result
            ]

            show_more = None
            for selector in show_more_selectors:
                show_more = await page.query_selector(selector)
                # Ensure it's visible
                if show_more and await show_more.is_visible():
                    break

            if not show_more:
                print("No visible 'Show More' button found. Stopping.")
                break

            try:
                # Scroll into view and click
                await show_more.scroll_into_view_if_needed()
                await show_more.click()
                click_count += 1
                print(f"Clicked 'Show More' #{click_count}, current listings: {current_count}")

                # Wait for the next listing to appear after click
                try:
                    await page.wait_for_selector(f"a.listing-card:nth-child({current_count+1})", timeout=20000)
                except:
                    print("No new listings detected after clicking. Stopping.")
                    break

            except Exception as e:
                print(f"Could not click 'Show More': {e}")
                break

        # Get final HTML after all listings are loaded
        content = await page.content()
        await browser.close()

    # Parse the listings
    soup = BeautifulSoup(content, "html.parser")
    listings = soup.select("a.listing-card")
    print(f"Total listings found: {len(listings)}")

    # Open aiohttp session for detail page mileage lookup
    async with aiohttp.ClientSession() as session:
        # Run detail page parsing concurrently for speed
        tasks = [parse_vehicle_listing(listing, session) for listing in listings]
        parsed_list = await asyncio.gather(*tasks)
        for parsed in parsed_list:
            if parsed:
                results.append(parsed)

    return results


async def save_to_db(records):
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        for r in records:
            await conn.execute(
                """
                INSERT INTO auctions (
                    url, title, year, make, model, original_owner,
                    mileage, sold_price, sold_date, status, excerpt
                )
                VALUES (
                    $1,$2,$3,$4,$5,$6,
                    $7,$8,$9,$10,$11
                )
                ON CONFLICT (url) DO NOTHING;
            """,
                r["url"],
                r["title"],
                r["year"],
                r["make"],
                r["model"],
                r["original_owner"],
                r["mileage"] if r["mileage"] is not None else None,  # NULL if missing
                r["sold_price"],
                r["sold_date"],
                r["status"],
                r["excerpt"],
            )
    finally:
        await conn.close()


async def main():
    url = "https://bringatrailer.com/porsche/997-gt3/"
    results = await scrape_bring_a_trailer(url)

    print(f"Scraped {len(results)} listings")
    if results:
        await save_to_db(results)
        print("Saved to database!")


if __name__ == "__main__":
    asyncio.run(main())
