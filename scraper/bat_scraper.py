import asyncio
import aiohttp
from aiohttp import ClientTimeout
import asyncpg
import re
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import google.generativeai as genai

from bat_util import (
    _ai_fallback_count,
    split_make_model,
    extract_mileage,
    extract_mileage_from_detail_page,
    extract_price_and_status,
    parse_sold_date,
    save_to_db,
)


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


async def scrape_bring_a_trailer(url: str, max_clicks: int = 400):
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
                # ".auctions-container.column-limited-width .auctions-footer-button",  # for all auction result
                ".auctions-footer .auctions-footer-content .auctions-footer-button",  # for all auction result
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
    if max_clicks == 200:
        listings.clear
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


async def main():
    url = "https://bringatrailer.com/honda/s2000/"
    results = await scrape_bring_a_trailer(url)

    # Log how many times AI fallback was used
    print(f"AI fallback used {_ai_fallback_count} times")

    print(f"Scraped {len(results)} listings")
    if results:
        await save_to_db(results)
        print("Saved to database!")


if __name__ == "__main__":
    asyncio.run(main())
