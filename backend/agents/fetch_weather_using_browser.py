"""Fetch official PAGASA weather pages that match a user's question.

Open-Meteo (see ``fetch_supabase_open_meteo_datasets.py``) gives model forecasts. This module
adds the official side: bulletins, advisories and flood information that are
only published as web pages on pagasa.dost.gov.ph.

Pages can be loaded two ways:

    * Headless browser - LangChain's ``AsyncChromiumLoader`` (Playwright).
      Slower, but runs the page's JavaScript. Requires a one-time
      ``uv run playwright install chromium``.
    * Plain HTTP request (httpx). Fast; PAGASA's bulletin and forecast pages
      already contain their text in the HTML, so this is enough today.

By default only the plain request is used: it is faster and cheaper, and
PAGASA's pages are static. Pass ``use_browser=True`` to render in Chromium
first (the plain request is still the fallback if the browser fails), for
example if PAGASA moves content behind JavaScript.

Example:
    pages = fetch_weather_pages("May bagyo ba ngayon?")
    pages[0]["source"]  # URL
    pages[0]["text"]    # cleaned page text, trimmed to MAX_TEXT_CHARS
"""

import logging

import httpx
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader

from utils.ttl_cache import ttl_cache

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; Flood-Gentic/0.1)"
REQUEST_TIMEOUT_SECONDS = 15.0
PAGE_CACHE_SECONDS = 30 * 60  # demo setting; bulletins are issued every few hours
MAX_TEXT_CHARS = 6000  # keeps the text small enough to hand to an LLM

# Official PAGASA pages, with the question words that make each one relevant.
PAGASA_PAGES = {
    "forecast": {
        "url": "https://www.pagasa.dost.gov.ph/weather",
        "keywords": [],  # default page when nothing else matches
    },
    "storm": {
        "url": "https://www.pagasa.dost.gov.ph/tropical-cyclone/severe-weather-bulletin",
        "keywords": ["typhoon", "bagyo", "storm", "cyclone", "signal", "habagat", "bulletin"],
    },
    "advisory": {
        "url": "https://www.pagasa.dost.gov.ph/weather/weather-advisory",
        "keywords": ["advisory", "warning", "rainfall warning", "thunderstorm", "babala"],
    },
    "flood": {
        "url": "https://www.pagasa.dost.gov.ph/flood",
        "keywords": ["flood", "baha", "river", "ilog", "dam"],
    },
}

# CSS selector for the main content area on PAGASA pages (first match wins).
CONTENT_SELECTOR = ".article-content, .tab-content"

# HTML tags that never hold useful weather text.
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "form", "noscript", "iframe"]


def choose_pages(question: str) -> list[str]:
    """Return the page keys (``storm``, ``flood``, ...) relevant to the question.

    Falls back to the general forecast page when no keyword matches.
    """
    text = question.lower()
    keys = [
        key
        for key, page in PAGASA_PAGES.items()
        if any(keyword in text for keyword in page["keywords"])
    ]
    return keys or ["forecast"]


def clean_html(html: str) -> tuple[str, str]:
    """Strip an HTML page down to ``(title, readable text)``."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # PAGASA pages put a long site menu before the real content, which would
    # use up the whole character limit. Read only the main content block when
    # the page has one; otherwise use the whole page.
    content = soup.select_one(CONTENT_SELECTOR) or soup

    # One line per block of text, blank lines removed.
    lines = (line.strip() for line in content.get_text(separator="\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return title, text[:MAX_TEXT_CHARS]


def load_html(url: str, use_browser: bool = False) -> str:
    """Return the raw HTML of a page.

    With ``use_browser`` the page is rendered in headless Chromium first; if
    that fails (for example Chromium is not installed) it falls back to a
    plain HTTP request. Without it, only the plain request is made.
    """
    if use_browser:
        try:
            documents = AsyncChromiumLoader([url], user_agent=USER_AGENT).load()
            if documents and documents[0].page_content.strip():
                return documents[0].page_content
        except Exception as error:  # Playwright raises several unrelated error types.
            logger.warning("Browser load failed for %s (%s); using plain request.", url, error)

    response = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.text


@ttl_cache(ttl_seconds=PAGE_CACHE_SECONDS)
def load_clean_page(url: str, use_browser: bool = False) -> tuple[str, str]:
    """Download a page and return ``(title, text)``. Cached for a few minutes.

    Every question about the same topic reuses one download, and the browser
    (the slowest option) only starts when the cache is empty.
    """
    return clean_html(load_html(url, use_browser=use_browser))


def fetch_weather_pages(question: str, use_browser: bool = False) -> list[dict]:
    """Fetch and clean the PAGASA pages that match the question.

    Returns one dict per page: ``{"page", "source", "title", "text"}``.
    A page that cannot be fetched is skipped (and logged) so one broken
    page does not lose the others.
    """
    results = []
    for key in choose_pages(question):
        url = PAGASA_PAGES[key]["url"]
        try:
            title, text = load_clean_page(url, use_browser)
        except Exception as error:
            logger.error("Could not fetch %s: %s", url, error)
            continue
        results.append({"page": key, "source": url, "title": title, "text": text})
    return results


if __name__ == "__main__":
    # Quick manual check: python -m agents.fetch_weather_using_browser "May bagyo ba?"
    import sys

    sample_question = " ".join(sys.argv[1:]) or "May bagyo ba ngayon?"
    for page in fetch_weather_pages(sample_question):
        print(f"== {page['title']} ({page['source']})")
        print(page["text"][:1500])
