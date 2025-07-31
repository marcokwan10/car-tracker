import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


def extract_price_and_status(text):
    match = re.search(r"(Sold for|Bid to) USD \$([\d,]+)", text)
    if match:
        status = "sold" if match.group(1) == "Sold for" else "reserve_not_met"
        price = int(match.group(2).replace(",", ""))
        return price, status
    return None, None


def extract_mileage_from_excerpt(excerpt: str):
    """
    Extracts the most likely total mileage from the excerpt.
    Prioritizes the highest number that appears to be the odometer reading.
    """
    # Common patterns: "12k miles", "now has 24k", "has 59k miles", "its 70k miles"
    mileage_patterns = re.findall(
        r"(?:(?:has|with|of|its|now has|approximately)?\s*)(\d{1,3}(?:[,\.]?\d{3})?k?)\s*miles?",
        excerpt,
        flags=re.IGNORECASE,
    )

    # Normalize and convert to integers
    mile_values = []
    for m in mileage_patterns:
        m_clean = m.lower().replace(",", "").replace(".", "")
        if "k" in m_clean:
            try:
                mile_values.append(int(float(m_clean.replace("k", "")) * 1000))
            except ValueError:
                continue
        else:
            try:
                mile_values.append(int(m_clean))
            except ValueError:
                continue

    if not mile_values:
        return None

    # Heuristic: return the highest mileage found
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
    miles_match = re.search(
        r"(\d{1,3}[,.]?\d{3}|[\d.]+[kK])\-?mile", title, re.IGNORECASE
    )
    if miles_match:
        miles_str = miles_match.group(1).replace(",", "").replace(".", "")
        if "k" in miles_match.group(1).lower():
            mileage = int(float(miles_str.replace("k", "")) * 1000)
        else:
            mileage = int(miles_str)
    else:
        mileage = extract_mileage_from_excerpt(excerpt)

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


async def scrape_bring_a_trailer(url: str, max_pages: int = 3):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        for _ in range(max_pages):
            load_more = await page.query_selector(".button-show-more")
            if load_more:
                await load_more.click()
                await page.wait_for_timeout(2000)
            else:
                break

        content = await page.content()
        await browser.close()

    soup = BeautifulSoup(content, "html.parser")
    listings = soup.select("a.listing-card")

    results = []

    async with aiohttp.ClientSession() as session:
        for listing in listings:
            parsed = await parse_vehicle_listing(listing, session)
            if parsed:
                results.append(parsed)

    return results


if __name__ == "__main__":
    import json

    url = "https://bringatrailer.com/ferrari/812-superfast/"
    results = asyncio.run(scrape_bring_a_trailer(url))
    print(json.dumps(results, indent=2))
