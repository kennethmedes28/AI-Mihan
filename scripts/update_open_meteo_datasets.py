"""Append the newest days from Open-Meteo to the two daily CSV files in datasets/.

Run from backend/ so the installed packages (pandas, httpx) are found:

    uv run python ../scripts/update_open_meteo_datasets.py --dry-run   # show the plan, call nothing
    uv run python ../scripts/update_open_meteo_datasets.py             # fetch and append
    uv run python ../scripts/update_open_meteo_datasets.py --end-date 2026-11-30

What it updates (the points come from datasets/ncr_locations.csv):

    ncr_weather_daily_2020-2026.csv   raw daily weather + river discharge
    ncr_flood_features_daily.csv      the same days plus rolling rain and discharge features

Only days after the last day already in the files are requested, and rows that
already exist keep their exact text, so running it twice in a row does nothing.
New rows are placed so each file stays sorted by location_id, then date.
Nothing is written unless every request succeeded.

Open-Meteo free-plan limits (non-commercial): 600 calls/minute, 5,000/hour,
10,000/day. A call over many days or many variables counts as several, so
requests are cut into short windows. Locations that share coordinates are
fetched once. Calls are spaced out, a "minutely" 429 waits and retries, and an
hourly or daily 429 stops the run with nothing written.

Known difference from the original data: every location in a coord_group is
requested at the same coordinates, so they get the same river discharge.
"""

import argparse
import time
from io import StringIO
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_DIR / "datasets"
LOCATIONS_CSV = DATASETS_DIR / "ncr_locations.csv"
WEATHER_CSV = DATASETS_DIR / "ncr_weather_daily_2020-2026.csv"
FEATURES_CSV = DATASETS_DIR / "ncr_flood_features_daily.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
TIMEZONE = "Asia/Manila"

# The newest days are still being finalised, so by default stop two days before today.
DEFAULT_END_LAG_DAYS = 2

# Rate limiting. 0.5 s between calls is at most 120 calls a minute (limit: 600).
SECONDS_BETWEEN_CALLS = 0.5
CHUNK_DAYS = 14  # longer ranges count as several calls against the limit
MAX_CALLS_PER_RUN = 500
MAX_RETRIES = 3
MINUTELY_LIMIT_WAIT_SECONDS = 61
REQUEST_TIMEOUT_SECONDS = 30.0

# Names in the API response -> column names in the weather CSV.
ARCHIVE_DAILY_COLUMNS = {
    "precipitation_sum": "precipitation_sum_mm",
    "temperature_2m_max": "temperature_2m_max_c",
    "temperature_2m_min": "temperature_2m_min_c",
    "shortwave_radiation_sum": "shortwave_radiation_sum_mjm2",
    "wind_speed_10m_max": "wind_speed_10m_max_kmh",
    "wind_gusts_10m_max": "wind_gusts_10m_max_kmh",
}
WET_DAY_MM = 1.0  # a day counts as wet from 1 mm of rain (matches the existing wet_streak_days)


class RateLimitError(RuntimeError):
    """Open-Meteo refused the request because an hourly or daily limit is used up."""


class OpenMeteoClient:
    """Small HTTP client that spaces out calls, retries politely and counts them."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls_made = 0
        self._last_call_time = 0.0
        self._http = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def get_json(self, url: str, params: dict) -> dict:
        """GET ``url`` and return the JSON body, waiting or retrying when Open-Meteo asks."""
        for attempt in range(1, MAX_RETRIES + 1):
            if self.calls_made >= self.max_calls:
                raise RateLimitError(f"Stopped after {self.max_calls} calls (--max-calls).")

            wait = SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()
            self.calls_made += 1

            response = self._http.get(url, params=params)
            if response.status_code == 200:
                return response.json()

            reason = self._error_reason(response)
            if response.status_code == 429:
                if "minutely" in reason.lower():
                    print(f"  minute limit reached, waiting {MINUTELY_LIMIT_WAIT_SECONDS}s")
                    time.sleep(MINUTELY_LIMIT_WAIT_SECONDS)
                    continue
                raise RateLimitError(f"Open-Meteo limit reached: {reason}. Try again later.")
            if response.status_code >= 500:
                time.sleep(2**attempt)  # server trouble: back off 2s, 4s, 8s
                continue
            raise RuntimeError(f"Open-Meteo returned {response.status_code}: {reason}")

        raise RuntimeError(f"Open-Meteo kept failing after {MAX_RETRIES} tries: {url}")

    @staticmethod
    def _error_reason(response: httpx.Response) -> str:
        """The error text Open-Meteo sends back, falling back to the raw body."""
        try:
            return str(response.json().get("reason", response.text))
        except ValueError:
            return response.text


def read_locations() -> pd.DataFrame:
    """Read the sample points (utf-8-sig drops the byte-order mark the file starts with)."""
    return pd.read_csv(LOCATIONS_CSV, encoding="utf-8-sig")


def find_first_missing_day(weather: pd.DataFrame, locations: pd.DataFrame) -> date:
    """The day after the oldest 'last day' among all locations (a new location starts at 2020-01-01)."""
    last_day = weather.groupby("location_id")["date"].max()
    first_missing = []
    for location_id in locations["location_id"]:
        if location_id in last_day.index:
            first_missing.append(date.fromisoformat(last_day[location_id]) + timedelta(days=1))
        else:
            first_missing.append(date(2020, 1, 1))
    return min(first_missing)


def split_into_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Cut start..end (inclusive) into windows of at most CHUNK_DAYS days."""
    windows = []
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((window_start, window_end))
        window_start = window_end + timedelta(days=1)
    return windows


def fetch_archive_window(client: OpenMeteoClient, latitude: float, longitude: float,
                         start: date, end: date) -> pd.DataFrame:
    """Daily weather plus the hourly-derived columns for one point and one window."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(ARCHIVE_DAILY_COLUMNS),
        "hourly": "temperature_2m,precipitation",
        "timezone": TIMEZONE,
    }
    body = client.get_json(ARCHIVE_URL, params)

    daily = pd.DataFrame(body["daily"]).rename(columns=ARCHIVE_DAILY_COLUMNS)
    daily = daily.rename(columns={"time": "date"})

    hourly = pd.DataFrame(body["hourly"])
    hourly["date"] = hourly["time"].str[:10]
    return daily.merge(summarise_hourly(hourly), on="date", how="left")


def summarise_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per day: hottest/coolest hour and the heaviest 1, 3 and 6 hour rain.

    The 3 and 6 hour totals are rolling sums inside the day (they do not
    cross midnight), which is how the existing files were built.
    """
    rows = []
    for day, hours in hourly.groupby("date"):
        rain = hours["precipitation"]
        rows.append({
            "date": day,
            "temperature_2m_max_from_hourly_c": hours["temperature_2m"].max(),
            "temperature_2m_min_from_hourly_c": hours["temperature_2m"].min(),
            "rain_max_1h_mm": rain.max(),
            "rain_max_3h_mm": rain.rolling(3).sum().max(),
            "rain_max_6h_mm": rain.rolling(6).sum().max(),
        })
    return pd.DataFrame(rows)


def fetch_discharge_window(client: OpenMeteoClient, latitude: float, longitude: float,
                           start: date, end: date) -> pd.DataFrame:
    """Daily river discharge for one point and one window."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "river_discharge",
    }
    body = client.get_json(FLOOD_URL, params)
    return pd.DataFrame(body["daily"]).rename(columns={"time": "date", "river_discharge": "river_discharge_m3s"})


def fetch_point(client: OpenMeteoClient, latitude: float, longitude: float,
                windows: list[tuple[date, date]]) -> pd.DataFrame:
    """All new days for one coordinate: weather, hourly-derived values and discharge."""
    pieces = []
    for start, end in windows:
        weather = fetch_archive_window(client, latitude, longitude, start, end)
        discharge = fetch_discharge_window(client, latitude, longitude, start, end)
        pieces.append(weather.merge(discharge, on="date", how="left"))
    point_days = pd.concat(pieces, ignore_index=True)

    # The newest days can come back empty while Open-Meteo is still computing them.
    return point_days.dropna(subset=["precipitation_sum_mm"])


def count_wet_streak(rain_mm: pd.Series) -> pd.Series:
    """Consecutive days (up to and including each day) with at least WET_DAY_MM of rain."""
    streaks = []
    current = 0
    for rain in rain_mm:
        current = current + 1 if rain >= WET_DAY_MM else 0
        streaks.append(current)
    return pd.Series(streaks, index=rain_mm.index)


def build_new_features(weather_all: pd.DataFrame, new_weather: pd.DataFrame,
                       hourly_columns: pd.DataFrame, locations: pd.DataFrame,
                       feature_columns: list[str]) -> pd.DataFrame:
    """Compute the features-file columns for the new rows.

    Rolling values need earlier days, so they are calculated on the whole
    series (old rows + new rows) and then only the new rows are kept.
    ``discharge_pctile`` ranks a day against that location's whole history.
    """
    elevation = locations.set_index("location_id")["elevation_m"]
    new_keys = set(zip(new_weather["location_id"], new_weather["date"]))
    result_pieces = []

    for location_id, series in weather_all.groupby("location_id"):
        series = series.sort_values("date").reset_index(drop=True)
        rain, discharge = series["precipitation_sum_mm"], series["river_discharge_m3s"]

        features = series[["location_id", "date", "precipitation_sum_mm", "wind_speed_10m_max_kmh",
                           "wind_gusts_10m_max_kmh", "river_discharge_m3s"]].copy()
        features["elevation_m"] = elevation[location_id]
        features["rain_3d_mm"] = rain.rolling(3, min_periods=1).sum().round(1)
        features["rain_7d_mm"] = rain.rolling(7, min_periods=1).sum().round(1)
        features["rain_14d_mm"] = rain.rolling(14, min_periods=1).sum().round(1)
        features["wet_streak_days"] = count_wet_streak(rain)

        # A zero yesterday would divide by zero; the existing file leaves those empty too.
        previous = discharge.shift(1).where(discharge.shift(1) > 0)
        features["discharge_change_1d_pct"] = ((discharge / previous - 1) * 100).round(1)
        features["discharge_max_3d_m3s"] = discharge.rolling(3, min_periods=1).max()
        features["discharge_pctile"] = discharge.rank(pct=True).round(3)

        is_new = [(location_id, day) in new_keys for day in features["date"]]
        result_pieces.append(features[is_new])

    new_features = pd.concat(result_pieces, ignore_index=True)
    new_features = new_features.merge(hourly_columns, on=["location_id", "date"], how="left")
    return new_features[feature_columns]


def merge_rows(csv_path: Path, rows: pd.DataFrame) -> None:
    """Add rows to a CSV, keeping it sorted by location_id then date.

    Old lines are read and written back as plain text, so their values are
    not reformatted; only the new lines are added in the right place.
    """
    old_text = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    new_text = pd.read_csv(StringIO(rows.to_csv(index=False)), dtype=str, keep_default_na=False)
    combined = pd.concat([old_text, new_text], ignore_index=True)
    combined = combined.sort_values(["location_id", "date"], key=sort_key, kind="stable")
    combined.to_csv(csv_path, index=False, lineterminator="\n")


def sort_key(column: pd.Series) -> pd.Series:
    """Sort location_id as a number (so 10 comes after 9); dates already sort as text."""
    return column.astype(int) if column.name == "location_id" else column


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--end-date", help="last day to fetch, YYYY-MM-DD (default: today minus 2 days)")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS_PER_RUN, help="stop if more calls are needed")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, make no requests, write nothing")
    args = parser.parse_args()

    locations = read_locations()
    weather = pd.read_csv(WEATHER_CSV, dtype={"date": str})
    weather_columns = list(weather.columns)
    feature_columns = list(pd.read_csv(FEATURES_CSV, nrows=0).columns)

    end = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=DEFAULT_END_LAG_DAYS)
    start = find_first_missing_day(weather, locations)
    if start > end:
        print(f"Already up to date: files end on {start - timedelta(days=1)}, requested end is {end}.")
        return

    windows = split_into_windows(start, end)
    points = locations.drop_duplicates(["latitude", "longitude"])[["latitude", "longitude"]]
    planned_calls = len(points) * len(windows) * 2  # one weather + one discharge call per window
    print(f"Missing days: {start} to {end} ({len(windows)} windows)")
    print(f"{len(points)} distinct coordinates for {len(locations)} locations -> {planned_calls} calls")
    if args.dry_run:
        return
    if planned_calls > args.max_calls:
        raise SystemExit(f"{planned_calls} calls needed but --max-calls is {args.max_calls}. "
                         "Raise it or use an earlier --end-date.")

    client = OpenMeteoClient(args.max_calls)
    try:
        days_by_coordinate = {}
        for point in points.itertuples():
            print(f"fetching {point.latitude}, {point.longitude}")
            days_by_coordinate[(point.latitude, point.longitude)] = fetch_point(
                client, point.latitude, point.longitude, windows)
    except RateLimitError as error:
        raise SystemExit(f"{error}\nNothing was written.")

    # Give every location the days of its coordinate, then drop days that already exist.
    existing_keys = set(zip(weather["location_id"], weather["date"]))
    new_pieces = []
    for location in locations.itertuples():
        days = days_by_coordinate[(location.latitude, location.longitude)].copy()
        days.insert(0, "location_id", location.location_id)
        keep = [(location.location_id, day) not in existing_keys for day in days["date"]]
        new_pieces.append(days[keep])
    new_days = pd.concat(new_pieces, ignore_index=True).sort_values(["location_id", "date"])

    if new_days.empty:
        print("Open-Meteo has no new complete days yet. Nothing written.")
        return

    new_weather = new_days[weather_columns].round({
        "precipitation_sum_mm": 1, "temperature_2m_max_c": 1, "temperature_2m_min_c": 1,
        "shortwave_radiation_sum_mjm2": 2, "wind_speed_10m_max_kmh": 1, "wind_gusts_10m_max_kmh": 1,
        "temperature_2m_max_from_hourly_c": 1, "temperature_2m_min_from_hourly_c": 1,
        "river_discharge_m3s": 2,
    })
    hourly_columns = new_days[["location_id", "date", "rain_max_1h_mm", "rain_max_3h_mm", "rain_max_6h_mm"]].round(1)
    new_features = build_new_features(pd.concat([weather, new_weather], ignore_index=True), new_weather,
                                      hourly_columns, locations, feature_columns)

    merge_rows(WEATHER_CSV, new_weather)
    merge_rows(FEATURES_CSV, new_features)
    print(f"Appended {len(new_weather)} rows ({new_weather['date'].min()} to {new_weather['date'].max()}) "
          f"to both files using {client.calls_made} calls.")


if __name__ == "__main__":
    main()
