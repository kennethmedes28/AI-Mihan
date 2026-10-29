"""Weather agent: answers weather questions from PAGASA and Open-Meteo.

It combines the two data sources written for this project:

    * ``fetch_weather_using_browser`` - official PAGASA bulletins and advisories
    * ``fetch_supabase_open_meteo_datasets`` - model forecast for the city in the question,
                                      plus its last few days of rain from Supabase

The two sources do not depend on each other, so they are fetched at the same
time. The slower one sets the wait, instead of the two waits adding up. If one
source fails, the answer is still built from the other.

The chat model is passed in, so this file does not depend on any one provider.

Example:
    from langchain_openai import ChatOpenAI  # or any LangChain chat model

    answer = answer_weather_question("Uulan ba bukas sa Makati?", ChatOpenAI())
"""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agents.fetch_supabase_open_meteo_datasets import fetch_open_meteo
from agents.fetch_weather_using_browser import fetch_weather_pages
from prompts.weather_prompt import WEATHER_PROMPT
from utils.call_budget import BUSY_ANSWER, model_budget
from utils.text import normalize_question
from utils.ttl_cache import MISSING, TTLCache
from utils.tracing import traced_agent

logger = logging.getLogger(__name__)

PAGASA_UNAVAILABLE = "PAGASA data is unavailable right now."
OPEN_METEO_UNAVAILABLE = "Open-Meteo data is unavailable right now."

# Fixed reply used when both sources are down (no model call, so no cost).
NO_DATA_ANSWER = (
    "I have no verified weather information right now. "
    "Check pagasa.dost.gov.ph or your local DRRMO.\n"
    "Wala akong na-verify na datos ng panahon ngayon. "
    "Tingnan ang pagasa.dost.gov.ph o makipag-ugnayan sa inyong lokal na DRRMO."
)

ANSWER_CACHE_SECONDS = 30 * 60  # demo setting
# Answers to repeated questions, kept briefly to avoid repeat model calls.
answer_cache = TTLCache(ttl_seconds=ANSWER_CACHE_SECONDS)

# Fewer characters sent to the model means fewer tokens billed.
MAX_PAGASA_CHARS_TOTAL = 2500
ALWAYS_KEEP_FIRST_CHARS = 700  # issue time and synopsis come first on PAGASA pages

# Lines containing these words are kept first when a page has to be shortened.
IMPORTANT_WORDS = [
    "signal", "warning", "advisory", "bulletin", "metro manila", "ncr", "manila",
    "rain", "flood", "wind", "gale", "thunderstorm", "storm", "issued",
]


# --- Fetching (both sources at once) -----------------------------------------


def fetch_both_sources(question: str) -> tuple[list[dict] | None, dict | None]:
    """Fetch PAGASA pages and Open-Meteo data in parallel.

    Returns ``(pagasa_pages, open_meteo_result)``. A source that fails is
    returned as ``None`` (and logged) so the other can still be used.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        # copy_context() carries the LangSmith run into the worker threads, so both fetches show up under this agent.
        pagasa_future = executor.submit(contextvars.copy_context().run, fetch_weather_pages, question)
        open_meteo_future = executor.submit(contextvars.copy_context().run, fetch_open_meteo, question)

        return _result_or_none(pagasa_future, "PAGASA"), _result_or_none(
            open_meteo_future, "Open-Meteo"
        )


def _result_or_none(future, source_name: str):
    """Return a future's result, or None if it raised an error."""
    try:
        return future.result()
    except Exception as error:
        logger.error("%s fetch failed: %s", source_name, error)
        return None


# --- Turning the data into prompt text ---------------------------------------


def shrink_text(text: str, budget: int, extra_words: list[str] | None = None) -> str:
    """Shorten text to about ``budget`` characters, keeping the useful lines.

    Kept, in their original order: the first lines (issue time, synopsis),
    then any line mentioning a warning, signal, wind, rain, flood, or one of
    ``extra_words`` such as the user's city. Everything else is dropped. This
    keeps safety-relevant lines that a plain cut at N characters could lose.
    """
    if len(text) <= budget:
        return text

    words = IMPORTANT_WORDS + [word.lower() for word in (extra_words or [])]
    lines = text.splitlines()

    keep = set()
    used = 0

    # 1. The opening lines.
    for index, line in enumerate(lines):
        if used >= ALWAYS_KEEP_FIRST_CHARS:
            break
        keep.add(index)
        used += len(line) + 1

    # 2. Important lines, until the budget is used up.
    for index, line in enumerate(lines):
        if index in keep:
            continue
        if any(word in line.lower() for word in words) and used + len(line) + 1 <= budget:
            keep.add(index)
            used += len(line) + 1

    return "\n".join(lines[index] for index in sorted(keep))[:budget]


def format_pagasa(pages: list[dict] | None, extra_words: list[str] | None = None) -> str:
    """Write the PAGASA pages as text for the prompt, within the character budget."""
    if not pages:
        return PAGASA_UNAVAILABLE

    # Share the budget between the pages, leaving room for each heading.
    per_page_budget = MAX_PAGASA_CHARS_TOTAL // len(pages)

    sections = []
    for page in pages:
        body = shrink_text(page["text"], per_page_budget, extra_words)
        sections.append(f"[{page['title']}] ({page['source']})\n{body}")

    return "\n\n".join(sections)


def format_open_meteo(result: dict | None) -> str:
    """Write the Open-Meteo current and daily values as short text lines.

    Hourly values are left out: they are long, and the daily summary is
    enough to answer most questions.
    """
    if not result:
        return OPEN_METEO_UNAVAILABLE

    weather = result["weather"]
    lines = [f"City: {result['plan']['city'].title()}"]

    current = weather.get("current", {})
    current_units = weather.get("current_units", {})
    current_values = [
        f"{name} {value}{current_units.get(name, '')}"
        for name, value in current.items()
        if name not in ("time", "interval")
    ]
    if current_values:
        lines.append(f"Now ({current.get('time', '?')}): " + ", ".join(current_values))

    daily = weather.get("daily", {})
    daily_units = weather.get("daily_units", {})
    for index, date in enumerate(daily.get("time", [])):
        values = [
            f"{name} {daily[name][index]}{daily_units.get(name, '')}"
            for name in daily
            if name != "time"
        ]
        lines.append(f"{date}: " + ", ".join(values))

    lines.extend(format_history(result.get("history")))
    return "\n".join(lines)


def format_history(history: list[dict] | None) -> list[str]:
    """Write the recent past days (from the Supabase dataset) as short text lines.

    Returns no lines when the history is missing, so the forecast is not affected.
    """
    if not history:
        return []

    lines = ["Recent days (observed, from the project dataset):"]
    for row in history:
        lines.append(
            f"{row['date']}: rain {row['rain_mm']}mm, 3-day {row['rain_3d_mm']}mm, "
            f"7-day {row['rain_7d_mm']}mm, river discharge {row['river_discharge_m3s']}m3/s"
        )
    return lines


# --- The agent -----------------------------------------------------------------


@traced_agent("weather_agent")
def answer_weather_question(question: str, llm: BaseChatModel) -> str:
    """Answer a weather question using PAGASA and Open-Meteo data.

    Cost savers:
        * The same question (ignoring case and punctuation) is answered from
          cache for a few minutes instead of calling the model again.
        * Source downloads are cached too (see the fetch modules).
        * If both sources are down, a fixed reply is returned and the model is
          not called. It is never cached, so the next request tries again.
        * The PAGASA text is shortened to the lines that matter.
    """
    cache_key = normalize_question(question)
    cached_answer = answer_cache.get(cache_key)
    if cached_answer is not MISSING:
        return cached_answer

    pagasa_pages, open_meteo_result = fetch_both_sources(question)
    if pagasa_pages is None and open_meteo_result is None:
        return NO_DATA_ANSWER

    # The detected city is used to keep PAGASA lines that mention it.
    city = open_meteo_result["plan"]["city"] if open_meteo_result else None

    if not model_budget.try_use():  # hourly cap on model calls (see utils/call_budget.py)
        return BUSY_ANSWER

    chain = WEATHER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": question,
            "pagasa_data": format_pagasa(pagasa_pages, [city] if city else None),
            "open_meteo_data": format_open_meteo(open_meteo_result),
        }
    )
    answer_cache.set(cache_key, answer)
    return answer


if __name__ == "__main__":
    # Quick manual check of the data side only (no model needed):
    #   python -m agents.weather_agent "Uulan ba bukas sa Makati?"
    import sys

    sample_question = " ".join(sys.argv[1:]) or "Will it rain tomorrow in Quezon City?"
    pagasa, open_meteo = fetch_both_sources(sample_question)
    print(format_open_meteo(open_meteo))
    print()
    print(format_pagasa(pagasa)[:1500])
