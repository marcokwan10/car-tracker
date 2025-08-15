import asyncio
from datetime import datetime
import json
import os
import re
import time
import random
from aiohttp import ClientTimeout
import aiohttp
import asyncpg
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

# --- Perf helpers ---
from time import perf_counter
from dataclasses import dataclass
from typing import Dict

# Load environment variables (for DB_* and GOOGLE_API_KEY)
load_dotenv()

# Configure Google Generative AI (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set; AI make/model fallback is disabled.")

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

# Detail page fetch tuning
DETAIL_MAX_RETRIES = int(os.getenv("DETAIL_MAX_RETRIES", "3"))
DETAIL_TIMEOUT = float(os.getenv("DETAIL_TIMEOUT", "20"))  # seconds
DETAIL_BACKOFF_BASE = float(os.getenv("DETAIL_BACKOFF_BASE", "1.8"))

DEFAULT_HEADERS = {
    "User-Agent": os.getenv("SCRAPER_USER_AGENT", "car-tracker/1.0 (+https://example.com)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

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
    "Plymouth",
    "Polestar",
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
    "Ducati",
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

# Sort by descending length so multi-word names are prioritized when matching
KNOWN_MAKES.sort(key=len, reverse=True)

# Normalize sub-brands or alternate names to canonical make
MAKE_NORMALIZATION = {
    "Mercedes-AMG": "Mercedes-Benz",
    "GMA": "Gordon Murray Automotive",
}


# ---------------- Performance instrumentation ----------------
@dataclass
class PerfStat:
    count: int = 0
    total: float = 0.0
    max: float = 0.0
    min: float = float("inf")


_perf_stats: Dict[str, PerfStat] = {}


def _record_perf(label: str, dt: float) -> None:
    st = _perf_stats.get(label)
    if st is None:
        st = PerfStat()
        _perf_stats[label] = st
    st.count += 1
    st.total += dt
    if dt > st.max:
        st.max = dt
    if dt < st.min:
        st.min = dt


class PerfTimer:
    def __init__(self, label: str):
        self.label = label
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _record_perf(self.label, perf_counter() - self._t0)


# Context helper: use as `with perf("phase.name"):`
def perf(label: str) -> PerfTimer:
    return PerfTimer(label)


def async_timed(label: str):
    """Decorator for async functions to record wall-clock duration per call."""

    def _wrap(fn):
        async def _inner(*args, **kwargs):
            t0 = perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _record_perf(label, perf_counter() - t0)

        return _inner

    return _wrap


def print_perf_summary() -> None:
    if not _perf_stats:
        print("=== Perf Summary ===\n(no samples)\n====================")
        return
    print("=== Perf Summary ===")
    for label, st in sorted(_perf_stats.items()):
        avg = (st.total / st.count) if st.count else 0.0
        print(f"{label:28s} n={st.count:5d} avg={avg*1000:.1f}ms min={st.min*1000:.1f}ms max={st.max*1000:.1f}ms")
    print("====================")


# ---------- Health check ----------
def log_health() -> None:
    """
    Print a one-line health summary at startup:
    - Whether Gemini is configured
    - Current RPM throttle & interval
    - Target DB host:port and database name
    """
    print("=== Startup Health Check ===")
    print(f"Gemini configured: {bool(GEMINI_API_KEY)}")
    if GEMINI_API_KEY:
        print(f"Gemini RPM limit: {_GEMINI_RATE_LIMIT_RPM} req/min | interval: {_gemini_interval:.4f}s")
    db_host = DB_CONFIG.get("host")
    db_port = DB_CONFIG.get("port")
    db_name = DB_CONFIG.get("database")
    print(f"DB target: {db_name}@{db_host}:{db_port}")
    print("============================")


# Cache AI results to avoid duplicate API calls
_ai_make_model_cache: dict[str, tuple[str, str] | None] = {}


# Rate limiting parameters for Gemini (requests per minute)
_GEMINI_RATE_LIMIT_RPM = int(os.getenv("GEMINI_RATE_LIMIT_RPM", "1200"))  # safe default if unset
_gemini_interval = 60.0 / max(1, _GEMINI_RATE_LIMIT_RPM)
_gemini_last_call = 0.0


def get_gemini_rpm() -> int:
    return _GEMINI_RATE_LIMIT_RPM


def suggest_ai_concurrency(target_latency_s: float = 0.25, overprovision: float = 1.5, cap: int = 128) -> int:
    """
    Suggest a reasonable asyncio concurrency for AI calls given the RPM throttle.
    Formula: concurrency ≈ RPS * target_latency * overprovision, clamped to [8, cap].
    With RPM=4000 (≈66.7 rps) and 250ms latency, this yields ~25; with 1.5x overprovision ≈ 38.
    """
    rps = get_gemini_rpm() / 60.0
    conc = int(max(8, min(cap, rps * max(0.05, target_latency_s) * max(1.0, overprovision))))
    return conc


# Count how many times AI fallback is invoked
_ai_fallback_count = 0


def get_ai_fallback_count() -> int:
    return _ai_fallback_count


def reset_ai_fallback_count() -> None:
    global _ai_fallback_count
    _ai_fallback_count = 0


async def _throttle_gemini_call():
    """
    Ensures we do not exceed the configured Gemini RPM by spacing calls.
    """
    global _gemini_last_call
    now = time.monotonic()
    wait = _gemini_interval - (now - _gemini_last_call)
    if wait > 0:
        await asyncio.sleep(wait)
    _gemini_last_call = time.monotonic()


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
    global _ai_fallback_count
    _ai_fallback_count += 1
    # Early exit if Gemini isn’t configured
    if not GEMINI_API_KEY:
        _ai_make_model_cache[raw_title] = (None, None)
        return None, None
    # Return cached result if available
    if raw_title in _ai_make_model_cache:
        return _ai_make_model_cache[raw_title] or (None, None)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
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
        await _throttle_gemini_call()
        response = await model.generate_content_async(prompt)
        # Parse JSON from response text
        parsed = json.loads(response.text)
        # If it's a list, take the first element
        if isinstance(parsed, list) and parsed:
            data = parsed[0]
        elif isinstance(parsed, dict):
            data = parsed
        else:
            data = {}
        result_tuple = (data.get("make"), data.get("model"))
        _ai_make_model_cache[raw_title] = result_tuple
        return result_tuple
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        _ai_make_model_cache[raw_title] = (None, None)
        return None, None


# ---------------- Gemini Transmission JSON Robust Parser -----------------
def _parse_transmission_from_model_response(text: str) -> str | None:
    """Return 'automatic' | 'manual' | None from a model text response.
    Tries strict JSON first, then extracts the first JSON-looking block, then
    falls back to keyword sniffing or bare string labels.
    """
    if not text:
        return None

    s = text.strip().strip("`")

    # 1) Direct JSON parse (object or bare string)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            val = obj.get("transmission")
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("automatic", "manual"):
                    return v
        elif isinstance(obj, str):
            v = obj.strip().lower()
            if v in ("automatic", "manual"):
                return v
    except Exception:
        pass

    # 2) Extract first {...} slice and try again
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                val = obj.get("transmission")
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in ("automatic", "manual"):
                        return v
        except Exception:
            pass

    # 3) Bare word responses
    if re.fullmatch(r'"?automatic"?', s, re.IGNORECASE):
        return "automatic"
    if re.fullmatch(r'"?manual"?', s, re.IGNORECASE):
        return "manual"

    # 4) Keyword sniffing (only if exactly one appears)
    has_auto = re.search(r"\bautomatic\b", s, re.IGNORECASE) is not None
    has_manual = re.search(r"\bmanual\b", s, re.IGNORECASE) is not None
    if has_auto ^ has_manual:
        return "automatic" if has_auto else "manual"

    return None


@async_timed("ai.identify_transmission")
async def ai_identify_transmission(raw_title: str, excerpt: str) -> bool | None:
    """
    Use Google Gemini to identify if the vehicle is automatic or manual.
    Returns True for automatic, False for manual, None if not identified.
    """
    # Early exit if Gemini isn’t configured
    if not GEMINI_API_KEY:
        return None

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config={"response_mime_type": "application/json"},
    )

    prompt = (
        "You are an expert vehicle information extractor.\n"
        "Determine whether the vehicle described below has an automatic or manual transmission.\n"
        "Return exactly one of these JSON objects: "
        '{"transmission":"automatic"} or {"transmission":"manual"}.\n'
        'If unknown, return {"transmission":"unknown"}.\n\n'
        f'Title: "{raw_title}"\n'
        f'Excerpt: "{excerpt}"'
    )

    # Up to 3 attempts on transient failures or malformed JSON
    for attempt in range(3):
        try:
            await _throttle_gemini_call()
            response = await model.generate_content_async(prompt)
            text = getattr(response, "text", None) or ""
            label = _parse_transmission_from_model_response(text)
            if label == "automatic":
                return True
            if label == "manual":
                return False
            # Unknown/malformed → retry with backoff (except after last attempt)
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            return None
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            print(f"An error occurred during Gemini API call: {e}")
            return None


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
    Uses limited retries with backoff and identifies only list items that mention miles.
    """
    html = None
    for attempt in range(DETAIL_MAX_RETRIES):
        try:
            async with session.get(
                url,
                timeout=ClientTimeout(total=DETAIL_TIMEOUT),
                headers=DEFAULT_HEADERS,
            ) as response:
                # Handle transient server / rate limiting responses
                if response.status in (429, 500, 502, 503, 504):
                    delay = min(DETAIL_BACKOFF_BASE**attempt, 10.0) + random.uniform(0, 0.5)
                    if attempt == DETAIL_MAX_RETRIES - 1:
                        print(f"Skipping detail page due to HTTP {response.status}: {url}")
                        return None
                    await asyncio.sleep(delay)
                    continue
                if response.status == 403:
                    # Might be a temporary block; short delay then retry
                    delay = 2.0 + attempt
                    if attempt == DETAIL_MAX_RETRIES - 1:
                        print(f"Skipping detail page due to HTTP 403: {url}")
                        return None
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                html = await response.text()
                break
        except aiohttp.client_exceptions.TooManyRedirects:
            print(f"Skipping detail page due to redirect loop: {url}")
            return None
        except asyncio.TimeoutError:
            if attempt == DETAIL_MAX_RETRIES - 1:
                print(f"Skipping detail page due to timeout: {url}")
                return None
            await asyncio.sleep(min(DETAIL_BACKOFF_BASE**attempt, 8.0))
        except aiohttp.ClientError as e:
            if attempt == DETAIL_MAX_RETRIES - 1:
                print(f"Skipping detail page due to client error: {e}: {url}")
                return None
            await asyncio.sleep(min(DETAIL_BACKOFF_BASE**attempt, 8.0))

    if not html:
        return None

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
        if "mile" not in text.lower():
            continue
        mileage = extract_mileage(text)
        if mileage is not None:
            return mileage

    return None


def parse_sold_date(date_str):
    if not date_str:
        return None
    for fmt in ["%m/%d/%y", "%b %d, %Y"]:  # handle "8/1/25" and "Aug 1, 2025"
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


async def save_to_db(records):
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        for r in records:
            await conn.execute(
                """
                INSERT INTO auction (
                    source, source_listing_id, url, title, year, make, model, original_owner,
                    mileage, sold_price, sold_date, status, excerpt, manual
                )
                VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,
                    $9,$10,$11,$12,$13,$14
                )
                ON CONFLICT (source, source_listing_id) DO UPDATE
                SET url=$3,
                    title=$4,
                    year=$5,
                    make=$6,
                    model=$7,
                    original_owner=$8,
                    mileage=COALESCE(EXCLUDED.mileage, auction.mileage),
                    sold_price=COALESCE(EXCLUDED.sold_price, auction.sold_price),
                    sold_date=COALESCE(EXCLUDED.sold_date, auction.sold_date),
                    status=COALESCE(EXCLUDED.status, auction.status),
                    excerpt=EXCLUDED.excerpt,
                    manual=COALESCE(EXCLUDED.manual, auction.manual);
            """,
                r.get("source"),
                r.get("source_listing_id"),
                r.get("url"),
                r.get("title"),
                r.get("year"),
                r.get("make"),
                r.get("model"),
                r.get("original_owner"),
                r.get("mileage") if r.get("mileage") is not None else None,
                r.get("sold_price"),
                r.get("sold_date"),
                r.get("status"),
                r.get("excerpt"),
                r.get("manual"),
            )
    finally:
        await conn.close()
