import asyncio
import aiohttp
import asyncpg
import re
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv

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


def parse_mileage_string(raw: str) -> int | None:
    """
    Converts a raw mileage string (like '19K', '19,500', '19.5k', '19k-Mile') to an integer mileage.
    Returns None if it cannot be parsed.
    """
    if not raw:
        return None

    # Normalize to lowercase, remove spaces and commas
    s = raw.lower().replace(",", "").replace(" ", "")
    s = s.replace("-mile", "").replace("mile", "")  # remove '-mile' or 'mile'

    # Regex to capture patterns like 19k, 19.5k, 19500
    match = re.match(r"(\d+(?:\.\d+)?)(k)?$", s)
    if not match:
        return None

    num = float(match.group(1))
    if match.group(2):  # has 'k'
        return int(num * 1000)
    return int(num)


def extract_mileage(text: str) -> int | None:
    """
    Scans a title or excerpt for the *most likely odometer reading*.
    Returns the highest mileage found, assuming that's the car's total mileage.
    """
    if not text:
        return None

    # Find mileage patterns in the text
    mileage_candidates = re.findall(
        r"(\d{1,3}(?:[.,]\d{3})?|[\d.]+k)\s*(?:-?mile|miles)?",
        text,
        flags=re.IGNORECASE,
    )

    mile_values = []
    for candidate in mileage_candidates:
        miles = parse_mileage_string(candidate)
        if miles is not None:
            mile_values.append(miles)

    if not mile_values:
        return None

    # Heuristic: total odometer is usually the highest number mentioned
    return max(mile_values)


async def extract_mileage_from_detail_page(session: aiohttp.ClientSession, url: str):
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

    for li in ul.find_all("li"):
        text = li.get_text(strip=True)
        match = re.search(r"(\d{1,3}(?:[,\.]?\d{3})?k?)\s*miles", text, re.IGNORECASE)
        if match:
            mileage_str = match.group(1).lower().replace(",", "").replace(".", "")
            if "k" in mileage_str:
                return int(float(mileage_str.replace("k", "")) * 1000)
            else:
                return int(mileage_str)

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

    # Try to extract mileage from title/excerpt
    mileage = extract_mileage(title) or extract_mileage(excerpt)

    # If still no mileage, fetch from detail page
    if mileage is None and url:
        try:
            mileage = await extract_mileage_from_detail_page(session, url)
        except Exception as e:
            print(f"Failed to fetch detail page for {url}: {e}")

    result_el = listing.select_one(".item-results")
    result_text = result_el.text.strip() if result_el else ""
    result_date = (
        result_el.select_one("span").text.strip().replace("on ", "")
        if result_el and result_el.select_one("span")
        else None
    )

    price, status = extract_price_and_status(result_text)

    return {
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
        "url": url,
    }


async def scrape_bring_a_trailer(url: str, max_clicks: int = 20):
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
                # New footer button
                ".auctions-container.column-limited-width .auctions-footer-button",
                # ".auctions-completed.page-section .auctions-footer-button",
                # Legacy button style
                # ".auctions-container.column-limited-width .button-show-more",
                # ".auctions-completed.page-section .button-show-more",
            ]

            show_more = None
            for selector in show_more_selectors:
                show_more = await page.query_selector(selector)
                if show_more:
                    break

            if not show_more:
                print("No more 'Show More' button found.")
                break

            try:
                await show_more.click()
                click_count += 1
                print(f"Clicked 'Show More' #{click_count}, current listings: {current_count}")

                # Wait for new items to load by checking if the count increases
                for _ in range(30):  # up to 30 * 0.5s = 15 seconds max
                    await page.wait_for_timeout(500)
                    new_count = len(await page.query_selector_all("a.listing-card"))
                    if new_count > current_count:
                        # New listings loaded, break inner loop
                        break
                else:
                    print("No new listings detected, stopping.")
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
        for listing in listings:
            parsed = await parse_vehicle_listing(listing, session)
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
                r["mileage"],
                r["sold_price"],
                r["sold_date"],
                r["status"],
                r["excerpt"],
            )
    finally:
        await conn.close()


async def main():
    url = "https://bringatrailer.com/auctions/results/"
    results = await scrape_bring_a_trailer(url)
    print(results[0])
    print(results[-1])
    print(f"Scraped {len(results)} listings")
    # if results:
    #     await save_to_db(results)
    #     print("Saved to database!")


if __name__ == "__main__":
    asyncio.run(main())
