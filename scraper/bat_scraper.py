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
    get_ai_fallback_count,
    split_make_model,
    extract_mileage,
    extract_mileage_from_detail_page,
    extract_price_and_status,
    parse_sold_date,
    save_to_db,
    log_health,
)


def _extract_source_listing_id(listing_tag) -> str | None:
    """Extract BaT's numeric listing ID from the list-card anchor.
    Tries `data-pusher="post;list;<id>"` first, then falls back to `data-watch-url` query param `listing=<id>`.
    """
    # 1) data-pusher="post;list;94414872"
    pusher = listing_tag.get("data-pusher")
    if pusher:
        parts = pusher.split(";")
        if parts and parts[-1].isdigit():
            return parts[-1]

    # 2) data-watch-url contains ...listing=<id>
    watch_el = listing_tag.select_one("[data-watch-url]")
    if watch_el:
        watch_url = watch_el.get("data-watch-url", "")
        m = re.search(r"[?&]listing=(\d+)", watch_url)
        if m:
            return m.group(1)

    return None


async def _collect_card_hrefs(page):
    """Return a set of absolute hrefs for visible listing cards under Past Auctions."""
    return set(
        await page.evaluate(
            """() => {
                const root = document.querySelector('.auctions-completed.page-section') || document;
                const as = Array.from(root.querySelectorAll('a.listing-card[href]'));
                const hrefs = [];
                for (const a of as) {
                    try { hrefs.push(new URL(a.href, location.href).href); } catch {}
                }
                return Array.from(new Set(hrefs));
            }"""
        )
    )


async def _get_show_more_btn(page):
    """Locate the Past Auctions Show More button and return its Locator or None."""
    selector = ".auctions-completed.page-section div.items-more button.button-show-more"
    loc = page.locator(selector)
    cnt = await loc.count()
    for i in range(min(cnt, 3)):
        btn = loc.nth(i)
        if await btn.is_visible():
            return btn
    return None


async def parse_vehicle_listing(listing, session: aiohttp.ClientSession):
    title_el = listing.select_one("h3")
    title = title_el.text.strip() if title_el else ""
    excerpt_el = listing.select_one(".item-excerpt")
    excerpt = excerpt_el.text.strip() if excerpt_el else ""
    url = listing.get("href")

    # --- Source & source ID (for DB natural key) ---
    source = "bat"
    source_listing_id = _extract_source_listing_id(listing)

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
        "source": source,
        "source_listing_id": source_listing_id,
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
        await page.set_viewport_size({"width": 1280, "height": 2000})
        await page.goto(url, timeout=60000)
        # Try to jump to Past Auctions tab/anchor so the right container renders
        try:
            tab = page.locator("a[href$='#past-auctions']").first
            if await tab.count() > 0:
                await tab.click()
                await page.wait_for_timeout(300)
            else:
                await page.evaluate(
                    "document.getElementById('past-auctions')?.scrollIntoView({behavior:'instant',block:'start'})"
                )
                await page.wait_for_timeout(200)
        except Exception:
            pass

        # Track unique card hrefs to detect real growth even if the DOM reorders
        prev_hrefs = await _collect_card_hrefs(page)

        click_count = 0

        while click_count < max_clicks:
            # Ensure lazy containers render
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(250)

            btn = await _get_show_more_btn(page)
            if not btn:
                print("No visible 'Show More' button found. Stopping.")
                break

            prev_count = len(prev_hrefs)

            try:
                # Click once
                await btn.scroll_into_view_if_needed()
                await btn.click()
                click_count += 1
                print(f"Clicked 'Show More' #{click_count}, previous listings: {prev_count}")

                # Observe loading text flip if present (best-effort)
                loading_span = page.locator(
                    ".auctions-completed.page-section div.items-more button.button-show-more span",
                    has_text="Loading more auctions",
                )
                try:
                    await loading_span.wait_for(state="visible", timeout=3000)
                except Exception:
                    pass
                try:
                    await loading_span.wait_for(state="hidden", timeout=12000)
                except Exception:
                    pass

                # Wait for the listings API/network to return (best-effort)
                try:
                    await page.wait_for_response(lambda r: "listings-filter" in r.url and r.ok, timeout=12000)
                except Exception:
                    pass

                # Poll for DOM growth: new count or new href
                increased = False
                for _ in range(60):  # ~15s at 250ms
                    await page.wait_for_timeout(250)
                    hrefs = await _collect_card_hrefs(page)
                    if len(hrefs) > prev_count or len(hrefs - prev_hrefs) > 0:
                        prev_hrefs = hrefs
                        print(f"Listings increased to {len(prev_hrefs)}")
                        increased = True
                        break
                    # keep nudging bottom to trigger lazy-loads
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

                if increased:
                    continue

                # Second chance: reacquire button and click again once
                btn = await _get_show_more_btn(page)
                if not btn:
                    print("Show More button disappeared; stopping.")
                    break
                await btn.scroll_into_view_if_needed()
                await btn.click()
                print("No increase detected; retried click once.")

                for _ in range(60):
                    await page.wait_for_timeout(250)
                    hrefs = await _collect_card_hrefs(page)
                    if len(hrefs) > prev_count or len(hrefs - prev_hrefs) > 0:
                        prev_hrefs = hrefs
                        print(f"Listings increased to {len(prev_hrefs)}")
                        increased = True
                        break
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

                if not increased:
                    print("No new listings detected after clicking (even after retry). Stopping.")
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

    connector = aiohttp.TCPConnector(limit=int(os.getenv("DETAIL_HTTP_LIMIT", "12")))
    timeout = ClientTimeout(total=float(os.getenv("DETAIL_SESSION_TIMEOUT", "30")))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(int(os.getenv("DETAIL_CONCURRENCY", "8")))

        async def _parse_with_sem(node):
            async with sem:
                return await parse_vehicle_listing(node, session)

        tasks = [_parse_with_sem(listing) for listing in listings]
        parsed_list = await asyncio.gather(*tasks)
        for parsed in parsed_list:
            if parsed:
                results.append(parsed)

    return results


async def main():
    # Health check: Gemini config, RPM throttle, DB target
    log_health()
    url = "https://bringatrailer.com/porsche/911/?yearTo=2000"
    results = await scrape_bring_a_trailer(url)

    # Log how many times AI fallback was used
    print(f"AI fallback used {get_ai_fallback_count()} times")

    print(f"Scraped {len(results)} listings")
    if results:
        await save_to_db(results)
        print("Saved to database!")


if __name__ == "__main__":
    asyncio.run(main())
