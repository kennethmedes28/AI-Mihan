"""Web lookup: the fallback used when the stored data cannot answer with confidence.

The document, weather and flood-risk agents each have their own idea of "low
confidence" (a weak document match, no PAGASA bulletin). When that happens they
call ``search_trusted_sites`` here. A headless browser (Chromium, via
``fetch_weather_using_browser.load_html``) opens the search page of each
trusted site, follows the best result links, and returns the cleaned page text.

Only the sites in ``TRUSTED_SITES`` are ever opened. Result links to any other
host are ignored, so a search page cannot send the browser somewhere else.

Answers built from this text are NOT verified documents. ``WEB_NOTE`` is added
to them so the user knows.

Example:
    pages = search_trusted_sites("Saan may bukas na evacuation center sa Tanza?")
    pages[0]["source"]  # URL
    pages[0]["text"]    # cleaned page text
"""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus, urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agents.fetch_weather_using_browser import clean_html, load_html
from prompts.document_prompt import DOCUMENT_PROMPT
from utils.call_budget import BUSY_ANSWER, model_budget
from utils.text import normalize_question
from utils.tracing import traced_agent
from utils.ttl_cache import MISSING, TTLCache

logger = logging.getLogger(__name__)

# The only sites the browser may open. ``search_url`` takes the question as ``{query}``.
TRUSTED_SITES = [
    {"domain": "ndrrmc.gov.ph", "search_url": "https://ndrrmc.gov.ph/?s={query}"},
    {"domain": "redcross.org.ph", "search_url": "https://redcross.org.ph/?s={query}"},
    {"domain": "dswd.gov.ph", "search_url": "https://www.dswd.gov.ph/?s={query}"},
]

LINKS_PER_SITE = 2  # result pages opened per site
MAX_QUERY_CHARS = 100
SEARCH_CACHE_SECONDS = 30 * 60  # demo setting
WEB_TEXT_CHARS = 1500  # per page, sent to the model

# Added to every answer that came from a web lookup.
WEB_NOTE = (
    "Note: this came from a web search of official sites, not our verified documents. "
    "Please confirm with your local DRRMO.\n"
    "Paalala: galing ito sa paghahanap sa opisyal na mga website, hindi sa aming na-verify na dokumento. "
    "Kumpirmahin sa inyong lokal na DRRMO."
)

# Search results, kept briefly. Empty results are not stored, so the next request tries again.
search_cache = TTLCache(ttl_seconds=SEARCH_CACHE_SECONDS)


def is_trusted_url(url: str, domain: str | None = None) -> bool:
    """True for an https link to a trusted site (or just to ``domain`` when given)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    domains = [domain] if domain else [site["domain"] for site in TRUSTED_SITES]
    return any(host == name or host.endswith("." + name) for name in domains)


def find_result_links(search_html: str, search_url: str, domain: str) -> list[str]:
    """Return the first few result links on a search page that stay on ``domain``."""
    soup = BeautifulSoup(search_html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        link = urljoin(search_url, anchor["href"]).split("#")[0]
        is_search_page = link.rstrip("/") == search_url.rstrip("/")
        if is_trusted_url(link, domain) and not is_search_page and link not in links:
            links.append(link)
        if len(links) == LINKS_PER_SITE:
            break
    return links


def search_one_site(site: dict, question: str) -> list[dict]:
    """Search one trusted site and return its best pages as ``{"page", "source", "title", "text"}``."""
    search_url = site["search_url"].format(query=quote_plus(question[:MAX_QUERY_CHARS]))
    try:
        search_html = load_html(search_url, use_browser=True)
    except Exception as error:  # Site down, blocked, Chromium missing, and so on.
        logger.warning("Web search failed on %s: %s", site["domain"], error)
        return []

    pages = []
    for link in find_result_links(search_html, search_url, site["domain"]):
        try:
            title, text = clean_html(load_html(link, use_browser=True))
        except Exception as error:
            logger.warning("Could not open %s: %s", link, error)
            continue
        if text.strip():
            pages.append({"page": "web", "source": link, "title": title, "text": text})
    return pages


@traced_agent("web_lookup", run_type="tool")
def search_trusted_sites(question: str) -> list[dict]:
    """Search every trusted site for the question. Returns an empty list if nothing was found."""
    cache_key = normalize_question(question)
    cached = search_cache.get(cache_key)
    if cached is not MISSING:
        return cached

    # The sites do not depend on each other, so search them at the same time.
    with ThreadPoolExecutor(max_workers=len(TRUSTED_SITES)) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, search_one_site, site, question)
            for site in TRUSTED_SITES
        ]
        pages = [page for future in futures for page in future.result()]

    if pages:
        search_cache.set(cache_key, pages)
    return pages


def format_web_sources(pages: list[dict]) -> str:
    """Write the pages as numbered sources for the document prompt."""
    return "\n\n".join(
        f"[{number}] (web | unverified | {page['source']})\n{page['text'][:WEB_TEXT_CHARS]}"
        for number, page in enumerate(pages, start=1)
    )


def answer_from_web(question: str, llm: BaseChatModel) -> str | None:
    """Answer a question from the trusted websites, or return None when they have nothing.

    The reply ends with ``WEB_NOTE``. The caller keeps its own "no information"
    reply for the None case.
    """
    pages = search_trusted_sites(question)
    if not pages:
        return None

    if not model_budget.try_use():  # hourly cap on model calls (see utils/call_budget.py)
        return BUSY_ANSWER

    chain = DOCUMENT_PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke({"question": question, "sources": format_web_sources(pages)})
    except Exception as error:  # Provider outage, rate limit, and so on.
        logger.error("Web answer failed: %s", error)
        return None
    return f"{answer}\n\n{WEB_NOTE}" if answer.strip() else None
