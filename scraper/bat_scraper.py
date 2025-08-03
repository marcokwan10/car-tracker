import asyncio
import aiohttp
import asyncpg
import re
import os
import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai

load_dotenv()  # Load .env file

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("FATAL: Please set the 'GEMINI_API_KEY' environment variable.")
    exit(1)

KNOWN_MAKES = [
    # Multi-word and longer names prioritized
    "Gordon Murray Automotive",
    "Mercedes-Benz",
    "Harley-Davidson",
    "BMW Motorrad",
    "Land Rover",
    "Morgan Aeromax",
    "Morgan SuperSport",
    "Rolls-Royce",
    "Mercedes-AMG",
    # Hypercar/hyper-specialist brands
    "Bugatti",
    "Koenigsegg",
    "Pagani",
    "Rimac",
    "Hennessey",
    "GMA",
    # Car manufacturers (current brands from Wikipedia)
    "Acura",
    "Abarth",
    "Alfa Romeo",
    "Alpina",
    "Alpine",
    "Aston Martin",
    "Audi",
    "Bentley",
    "BMW",
    "BYD",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Citroën",
    "Dacia",
    "Daewoo",
    "Daihatsu",
    "Dodge",
    "Donkervoort",
    "DS",
    "Ferrari",
    "Fiat",
    "Fisker",
    "Ford",
    "Genesis",
    "Honda",
    "Hummer",
    "Hyundai",
    "Infiniti",
    "Iveco",
    "Jaguar",
    "Jeep",
    "Kia",
    "KTM",
    "Lada",
    "Lamborghini",
    "Lancia",
    "Landwind",
    "Lexus",
    "Lucid",
    "Lotus",
    "Maserati",
    "Maybach",
    "Mazda",
    "McLaren",
    "Mercedes",
    "Mini",
    "Mitsubishi",
    "Morgan",
    "Nissan",
    "Opel",
    "Peugeot",
    "Plymouth" "Polestar",
    "Pontiac",
    "Porsche",
    "Ram",
    "Renault",
    "Rivian",
    "Rolls-Royce",
    "Rover",
    "Saab",
    "Saturn",
    "Scion",
    "Seat",
    "Skoda",
    "Smart",
    "SsangYong",
    "Subaru",
    "Suzuki",
    "Tata",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
    "International",
    "Mercury",
    "GMC",
    # Motorcycle manufacturers (major current)
    "Aprilia",
    "Benelli",
    "Bimota",
    "BMW Motorrad",
    "Ducati",
    "Harley-Davidson",
    "Hero MotoCorp",
    "Husqvarna",
    "Indian",
    "Kawasaki",
    "KTM",
    "Moto Guzzi",
    "MV Agusta",
    "Piaggio",
    "Royal Enfield",
    "Suzuki",
    "Triumph",
    "TVS",
    "Vespa",
    "Yamaha",
    "Zero Motorcycles",
    "BSA",
]

# Normalize sub-brands or alternate names to canonical make
MAKE_NORMALIZATION = {
    "Mercedes-AMG": "Mercedes-Benz",
    "GMA": "Gordon Murray Automotive",
}

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
}


async def split_make_model(raw_title: str) -> tuple[str, str]:
    """
    raw_title = e.g. "2021 Land Rover Range Rover Evoque"
    returns ("Land Rover", "Range Rover Evoque")
    """

    # Remove common non-make prefixes
    title = re.sub(
        r"^(One-Owner|Original-Owner|Modified|Supercharged|Turbocharged|Custom|JDM)\s+",
        "",
        raw_title,
        flags=re.IGNORECASE,
    )

    # strip off year and mileage
    # e.g. turn "15k-Mile 2021 Land Rover…" into "Land Rover Range Rover Evoque"
    stripped_title = re.sub(r"^(?:[\d.,kK-]+-Mile\s+)?(?:19|20)\d{2}\s+", "", title).strip()

    # find known make
    for make in KNOWN_MAKES:
        if stripped_title.startswith(make + " "):
            model = stripped_title[len(make) :].strip()
            normalized_make = MAKE_NORMALIZATION.get(make, make)
            return normalized_make, model

    # fallback to use ai if no known make was found
    result = await ai_extract_make_model(raw_title)
    return result if result else (None, None)


async def ai_extract_make_model(raw_title: str) -> tuple[str, str]:
    """
    Use Google Gemini to extract the vehicle make and model from the title.
    Only used as a fallback when deterministic extraction fails.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config={"response_mime_type": "application/json"},
    )

    prompt = f"""
    You are an expert vehicle information extractor.
    From the car title below, extract the make and the model.
    Return the output as a JSON object with two keys: "make" and "model".

    If the title does not seem to be for a vehicle, return a JSON object where both "make" and "model" are null.

    Title: "{raw_title}"
    """
    try:
        response = await model.generate_content_async(prompt)

        # The API's JSON mode directly provides the parsed JSON in the response text
        result = json.loads(response.text)

        if result is None:
            return None, None

        return result.get("make"), result.get("model")

    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return None, None


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

    # --- Extract year ---
    year_match = re.search(r"(?:19\d{2}|20\d{2})", title)
    year = int(year_match.group(0)) if year_match else None
    if not year:
        return None  # Skip listing with no year.

    # --- Hybrid make/model extraction ---
    make, model = await split_make_model(title)
    if not make or not model:
        return None  # Skip non-vehicle listings like wheels, hardtops, etc.

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


async def scrape_bring_a_trailer(url: str, max_clicks: int = 10):
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
    url = "https://bringatrailer.com/auctions/results/"
    results = await scrape_bring_a_trailer(url)

    print(f"Scraped {len(results)} listings")
    if results:
        await save_to_db(results)
        print("Saved to database!")


if __name__ == "__main__":
    asyncio.run(main())
