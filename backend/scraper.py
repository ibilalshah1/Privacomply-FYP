"""
Privacy Policy Scraper
======================
Given any website URL (e.g. "https://google.com"), this module:

  1. Normalises the URL (adds https:// if missing)
  2. Finds the privacy policy page:
       a. Probes ~20 common URL paths (fast HEAD requests)
       b. Scans homepage links with scored keyword matching
       c. Falls back to the given URL if nothing is found
  3. Fetches the page with httpx; retries with Playwright if JS-rendered
  4. If the URL resolves to a PDF, extracts text with pdfplumber / PyMuPDF
  5. Strips noise: nav, header, footer, scripts, banners, cookie notices
  6. Returns a PrivacyPolicyResult dataclass (cleaned text + metadata)

Public API
----------
  get_privacy_policy(url: str) -> PrivacyPolicyResult
      High-level convenience wrapper — the function requested by the task.

  scrape_privacy_policy(website_url: str) -> tuple[str, str]
      Legacy entry point preserved for backward compatibility.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─────────────────────────────────────────────
# Common privacy policy URL paths to probe first
# ─────────────────────────────────────────────
PRIVACY_PATHS = [
    "/privacy-policy",
    "/privacy",
    "/privacypolicy",
    "/privacy-notice",
    "/privacy-statement",
    "/legal/privacy",
    "/legal/privacy-policy",
    "/policies/privacy",
    "/policy/privacy",
    "/about/privacy",
    "/en/privacy",
    "/en/privacy-policy",
    "/terms/privacy",
    "/info/privacy",
    "/help/privacy",
    "/data-privacy",
    "/data-protection",
    "/gdpr",
    "/cookie-policy",
    "/legal",
]

# ─────────────────────────────────────────────
# Scored keyword sets for link discovery
# Higher score = more specific / trustworthy match
# ─────────────────────────────────────────────
_KEYWORD_SCORES: list[tuple[re.Pattern, int]] = [
    (re.compile(r"privacy[\s\-_]?policy",     re.I), 10),
    (re.compile(r"privacy[\s\-_]?notice",     re.I),  9),
    (re.compile(r"privacy[\s\-_]?statement",  re.I),  9),
    (re.compile(r"data[\s\-_]?privacy",       re.I),  8),
    (re.compile(r"data[\s\-_]?protection",    re.I),  7),
    (re.compile(r"data[\s\-_]?policy",        re.I),  7),
    (re.compile(r"\bprivacy\b",               re.I),  6),
    (re.compile(r"personal[\s\-_]?data",      re.I),  5),
    (re.compile(r"\blegal\b",                 re.I),  3),
    (re.compile(r"\bgdpr\b",                  re.I),  4),
    (re.compile(r"\bcookie[\s\-_]?policy\b",  re.I),  2),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# HTML tags that never contain policy text
_NOISE_TAGS = [
    "script", "style", "noscript", "iframe", "svg", "img",
    "nav", "header", "footer", "aside", "form", "button",
    "dialog", "template", "figure", "figcaption",
]

# CSS selectors for common noise elements
_NOISE_SELECTORS = [
    "[class*='cookie']", "[id*='cookie']",
    "[class*='banner']", "[id*='banner']",
    "[class*='popup']",  "[id*='popup']",
    "[class*='modal']",  "[id*='modal']",
    "[class*='toast']",
    "[class*='menu']",   "[id*='menu']",
    "[class*='navbar']", "[id*='navbar']",
    "[class*='sidebar']","[id*='sidebar']",
    "[class*='breadcrumb']",
    "[class*='social']",
    "[aria-label='breadcrumb']",
    "[class*='advertisement']",
    "[class*='promo']",
]


# ═══════════════════════════════════════════════
# Output dataclass
# ═══════════════════════════════════════════════

@dataclass
class PrivacyPolicyResult:
    """Structured result returned by get_privacy_policy()."""
    text: str                          # cleaned policy text
    url: str                           # final URL of the policy page
    source_type: str = "html"          # "html" | "pdf" | "playwright"
    word_count: int = 0
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())


# ═══════════════════════════════════════════════
# 1. URL HELPERS
# ═══════════════════════════════════════════════

def normalize_url(url: str) -> str:
    """Add https:// scheme if missing and strip trailing slashes."""
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _root(url: str) -> str:
    """Return scheme + netloc (e.g. 'https://example.com')."""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _is_pdf_url(url: str) -> bool:
    """Return True if the URL almost certainly points to a PDF file."""
    return urlparse(url).path.lower().endswith(".pdf")


def _is_pdf_response(resp: httpx.Response) -> bool:
    """Return True if the HTTP response contains a PDF."""
    ct = resp.headers.get("content-type", "").lower()
    return "application/pdf" in ct or resp.url.path.lower().endswith(".pdf")


# ═══════════════════════════════════════════════
# 2. LINK SCORING
# ═══════════════════════════════════════════════

def _score_link(text: str, href: str) -> int:
    """
    Score a hyperlink by how likely it is to be a privacy policy.
    Higher is better; 0 means not relevant.
    """
    score = 0
    combined = f"{text} {href}"
    for pattern, weight in _KEYWORD_SCORES:
        if pattern.search(combined):
            score += weight
    # Penalise very long paths — real policy pages have short URLs
    path_depth = href.count("/")
    score -= max(0, path_depth - 3)
    return score


def _best_privacy_link(soup: BeautifulSoup, base_url: str) -> str | None:
    """
    Scan all <a> tags, score them, and return the best candidate URL.
    Returns None if no sufficiently relevant link is found.
    """
    base = base_url.rstrip("/")
    scored: list[tuple[int, str]] = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(separator=" ", strip=True)
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue

        full_url = urljoin(base_url, href)
        # Skip same-page anchors
        if full_url.rstrip("/") == base:
            continue

        score = _score_link(text, href)
        if score > 0:
            scored.append((score, full_url))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_url = scored[0]
    logger.debug("Top link candidates: %s", scored[:5])

    # Require a minimum relevance threshold
    return best_url if best_score >= 3 else None


# ═══════════════════════════════════════════════
# 3. FIND PRIVACY POLICY URL
# ═══════════════════════════════════════════════

def find_privacy_url(website_url: str) -> str:
    """
    Locate the privacy policy URL for a given website.

    Strategy:
      1. Probe ~20 well-known URL paths with HEAD requests (fast).
      2. Download the homepage and score all links.
      3. Fall back to the original URL if nothing is found.
    """
    base = normalize_url(website_url)
    root = _root(base)

    # ── Step 1: probe common paths ────────────────
    for path in PRIVACY_PATHS:
        candidate = root + path
        try:
            r = httpx.head(
                candidate, headers=HEADERS, timeout=5,
                follow_redirects=True,
            )
            if r.status_code == 200:
                logger.info("Found privacy policy via path probe: %s", candidate)
                return str(r.url)   # use final URL after redirects
        except Exception:
            continue

    # ── Step 2: scan homepage links ───────────────
    logger.info("Path probes failed — scanning homepage: %s", base)
    try:
        r = httpx.get(base, headers=HEADERS, timeout=15, follow_redirects=True)
        soup = BeautifulSoup(r.content, "lxml")
        best = _best_privacy_link(soup, str(r.url))
        if best:
            logger.info("Found privacy link via homepage scan: %s", best)
            return best
    except Exception as exc:
        logger.warning("Homepage scan failed: %s", exc)

    # ── Step 3: fallback ──────────────────────────
    logger.warning("No privacy policy link found — using original URL: %s", base)
    return base


# ═══════════════════════════════════════════════
# 4. PDF EXTRACTION
# ═══════════════════════════════════════════════

def _extract_pdf_text(data: bytes) -> str:
    """
    Extract text from a PDF byte stream.
    Tries pdfplumber first (better layout), then PyMuPDF as fallback.
    Raises ImportError if neither library is available.
    """
    # ── Try pdfplumber ────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        if text.strip():
            logger.info("PDF extracted via pdfplumber (%d pages)", len(pages))
            return text
    except ImportError:
        logger.debug("pdfplumber not installed — trying PyMuPDF")
    except Exception as exc:
        logger.warning("pdfplumber extraction failed: %s", exc)

    # ── Try PyMuPDF (fitz) ────────────────────────
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(p for p in pages if p.strip())
        logger.info("PDF extracted via PyMuPDF (%d pages)", len(pages))
        return text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed: %s", exc)

    raise ImportError(
        "PDF privacy policy detected but no PDF library is installed.\n"
        "Fix: pip install pdfplumber   (or: pip install PyMuPDF)"
    )


# ═══════════════════════════════════════════════
# 5. HTML CLEANING
# ═══════════════════════════════════════════════

def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    Remove noise elements then extract text from the most likely
    main-content container, falling back to <body>.
    """
    # Remove noise tags entirely
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # Remove noise by CSS selector
    for sel in _NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    # Prefer semantic / ID-based content containers
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"(content|main|privacy|policy|body)", re.I))
        or soup.find(class_=re.compile(r"(content|main|privacy|policy)", re.I))
        or soup.find("body")
    )

    return (main or soup).get_text(separator="\n", strip=True)


def _clean_text(raw: str) -> str:
    """
    Post-process extracted plain text:
      - Drop lines shorter than 15 characters (navigation artefacts)
      - Remove duplicate lines (preserve order)
      - Collapse runs of blank lines to a single blank line
    """
    lines = raw.splitlines()
    seen: set[str] = set()
    kept: list[str] = []

    for line in lines:
        line = line.strip()
        if len(line) < 15:
            continue
        norm = re.sub(r"\s+", " ", line.lower())
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════
# 6. PLAYWRIGHT FALLBACK (JS-rendered pages)
# ═══════════════════════════════════════════════

def _fetch_with_playwright(url: str) -> str:
    """
    Render the page in a headless Chromium browser and return HTML.
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Page requires JavaScript but Playwright is not installed.\n"
            "Fix: pip install playwright && playwright install chromium"
        )

    logger.info("JS-rendered page — launching Playwright for: %s", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, wait_until="networkidle", timeout=30_000)
        page.wait_for_timeout(2_000)   # let lazy content load
        html = page.content()
        browser.close()
    return html


# ═══════════════════════════════════════════════
# 7. PUBLIC ENTRY POINTS
# ═══════════════════════════════════════════════

def get_privacy_policy(url: str) -> PrivacyPolicyResult:
    """
    High-level function: given any website URL, return its privacy policy text.

    Steps:
      1. Locate the privacy policy URL (path probing + scored link scan).
      2. Fetch the page with httpx.
      3. If the response is a PDF, extract text with pdfplumber / PyMuPDF.
      4. If the HTML yields too little text (JS-rendered), retry with Playwright.
      5. Clean the text and return a PrivacyPolicyResult.

    Parameters
    ----------
    url : str
        Homepage URL of the target website (e.g. "https://google.com").

    Returns
    -------
    PrivacyPolicyResult
        .text        — cleaned privacy policy text
        .url         — final URL of the policy page
        .source_type — "html" | "pdf" | "playwright"
        .word_count  — word count of the cleaned text
        .errors      — list of non-fatal warnings encountered
    """
    errors: list[str] = []
    source_type = "html"

    # ── Step 1: find the privacy policy URL ──────
    privacy_url = find_privacy_url(url)
    logger.info("Fetching privacy policy from: %s", privacy_url)

    # ── Step 2: fetch the page ───────────────────
    try:
        resp = httpx.get(
            privacy_url, headers=HEADERS, timeout=20, follow_redirects=True
        )
        resp.raise_for_status()
        final_url = str(resp.url)
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP {exc.response.status_code} fetching {privacy_url}"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", privacy_url, exc)
        raise

    # ── Step 3: PDF handling ──────────────────────
    if _is_pdf_response(resp):
        logger.info("Response is a PDF — extracting text")
        source_type = "pdf"
        raw_text = _extract_pdf_text(resp.content)
        text = _clean_text(raw_text)
        return PrivacyPolicyResult(
            text=text,
            url=final_url,
            source_type=source_type,
            errors=errors,
        )

    # ── Step 4: HTML extraction ───────────────────
    soup = BeautifulSoup(resp.content, "lxml")
    text = _clean_text(_extract_main_text(soup))
    word_count = len(text.split())
    logger.info("httpx extraction: %d words", word_count)

    # ── Step 5: Playwright fallback for JS pages ──
    if word_count < 100:
        warn = f"Only {word_count} words via httpx — retrying with Playwright"
        logger.warning(warn)
        errors.append(warn)
        try:
            html = _fetch_with_playwright(final_url)
            soup = BeautifulSoup(html, "lxml")
            text = _clean_text(_extract_main_text(soup))
            word_count = len(text.split())
            source_type = "playwright"
            logger.info("Playwright extraction: %d words", word_count)
        except Exception as exc:
            errors.append(f"Playwright fallback failed: {exc}")
            logger.error("Playwright fallback failed: %s", exc)

    # ── Step 6: quality guard ─────────────────────
    if len(text.split()) < 50:
        raise ValueError(
            f"Extracted only {len(text.split())} words from {final_url}. "
            "The page may be behind a login, CAPTCHA, or bot protection."
        )

    return PrivacyPolicyResult(
        text=text,
        url=final_url,
        source_type=source_type,
        errors=errors,
    )


# ─────────────────────────────────────────────
# Legacy entry point (backward compatibility)
# ─────────────────────────────────────────────

def scrape_privacy_policy(website_url: str) -> tuple[str, str]:
    """
    Legacy wrapper around get_privacy_policy().
    Returns (cleaned_policy_text, privacy_policy_url).
    """
    result = get_privacy_policy(website_url)
    return result.text, result.url
