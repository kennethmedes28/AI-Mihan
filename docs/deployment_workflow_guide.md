# Deployment and Workflow Guide

How Flood-Gentic is configured, run and kept up to date. The target architecture (Railway, Vercel, Supabase) is in
`docs/backend_architecture.md` section 9 and `docs/frontend_architecture.md` section 8; this guide covers what exists today.

## 1. What is automated today

| What | How | Where |
|---|---|---|
| Daily weather datasets (CSV files and Supabase tables) | GitHub Actions, every day at 00:00 Philippine time | `.github/workflows/update-datasets.yml`, details in `docs/dataset_updates.md` |
| Everything else (corpus ingestion, FAISS runs, location chunks) | Run by hand from `scripts/` | section 4 below |

The Railway API service, the ingestion worker and the Vercel frontend are planned but not set up: `backend/main.py` is still a stub.

## 2. Configuration

Secrets live in `backend/.env` (gitignored; never commit it). Variable names only:

| Variable | Used for |
|---|---|
| `SUPABASE_PROJECT_REF`, `SUPABASE_PASSWORD`, `SUPABASE_POOLER_HOST` | Postgres connection through the Supabase transaction pooler (port 6543). `SUPABASE_DB_URL` can be set instead as one connection string |
| `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY`, `SUPABASE_ACCESS_TOKEN` | Supabase REST and management access; not used by the current code. The secret key bypasses row-level security, so keep it out of the browser |
| `MUSE_API` | Muse Spark chat models and the Muse speech-to-text (`api.meta.ai`) |
| `ORS_API_KEY` | OpenRouteService routing for the route agent |
| `HF_TOKEN`, `EMBEDDING_MODEL` | Downloading and choosing the embedding model |
| `DEPTH_MODEL` | Optional: the metric depth model for the flood depth estimate (default Depth Anything V2 Metric Outdoor Small, about 100 MB, downloaded on first use) |
| `LANGSMITH_KEY`, `LANGSMITH_TRACE`, `LANGSMITH_PROJECT` | LangSmith tracing, see section 6 |
| `GROQ_API_KEY`, `LLAMAINDEX_API_KEY` | Other model providers |

`backend/.env.example` lists the Supabase names, but the `.env.*` rule in `.gitignore` also ignores it, so it is not in the repository.
Remove that rule's effect for the example file (for example with `!.env.example`) if you want it shared.

## 3. Daily dataset workflow

`update-datasets.yml` runs `cron: "0 16 * * *"` (16:00 UTC = 00:00 Philippine time, no daylight saving) and can be started by hand
with **Run workflow**. GitHub may start it a few minutes late.

Steps: install `pandas`, `httpx`, `psycopg[binary]`, `python-dotenv` → `scripts/update_open_meteo_datasets.py` (fetch missing days)
→ `scripts/sync_datasets_to_supabase.py` (append only the new rows) → commit the two CSV files back to the branch.

Set up once, in the repository's Settings, Secrets and variables, Actions: `SUPABASE_PROJECT_REF`, `SUPABASE_PASSWORD`, `SUPABASE_POOLER_HOST`.
The commit step needs `contents: write` (already set in the file); if branch protection blocks the push, allow the Actions bot to
bypass it or remove that step. If the Supabase step fails nothing is committed, so the next run fetches the same days again.
Check a run under the repository's Actions tab; the log shows how many rows were added to each table.

## 4. Manual jobs (run from `backend/`)

| When | Command | Notes |
|---|---|---|
| First setup, or a full reload of the datasets | `uv run python ../scripts/load_to_supabase.py` | Empties and reloads the three dataset tables, upserts the FAISS run's chunks. `--dry-run` checks the files only |
| A PDF or the emergency contacts changed | `uv run python ../scripts/ingest_corpus.py --supabase` | Re-embeds the whole corpus (slow, loads the embedding model) and saves a new FAISS run |
| `corpus/ncr_locations.csv` changed | `uv run python ../scripts/refresh_location_chunks.py` | Re-embeds only the 17 location chunks, updates Supabase and saves a new FAISS run |
| `datasets/ncr_locations.csv` changed | `uv run python ../scripts/update_open_meteo_datasets.py --rebuild`, then `load_to_supabase.py` | `location_id` is the key in both daily files and Supabase, so the history must be rebuilt (about 600 API calls, 20 to 30 minutes; rerun the same command to resume after an error) |
| Catch up the datasets by hand | `uv run python ../scripts/update_open_meteo_datasets.py`, then `sync_datasets_to_supabase.py` | Same as the workflow; `--dry-run` shows the plan |

Both `locations` files (`corpus/` and `datasets/`) currently hold the same 17 city points. Change them together.

## 5. Checks before and after a change

- Tests: `uv run --with pytest python -m pytest tests -q` from `backend/` (39 pass at the time of writing).
- Dataset files: both daily CSVs should have the same rows, be sorted by `location_id` then `date`, and have no duplicate `(location_id, date)`.
- Supabase: row counts of `ncr_weather_daily` and `ncr_flood_features_daily` match the CSVs; `chunks` matches the newest FAISS run (359 chunks).
- The weather agent's recent history: `fetch_recent_history("makati")` returns 7 rows.

## 6. LangSmith tracing

Tracing records every chat message as a run in LangSmith so slow or failing agents can be found. It is on when `backend/.env` has
`LANGSMITH_KEY` and `LANGSMITH_TRACE=true`; `LANGSMITH_PROJECT` names the project (default `flood-gentic`). The LangSmith library itself reads
`LANGSMITH_API_KEY` and `LANGSMITH_TRACING`, so `utils/tracing.py` copies the values over when the first agent is imported. Setting
`LANGSMITH_TRACE=false` (or removing the key) turns it off; nothing else changes.

- **What a trace looks like:** `flood-gentic-chat` (tag `flood-gentic`, metadata `thread_id` and `has_gps`) contains `route_message`, the chosen node and,
  inside it, the agent run (`contacts_agent`, `weather_agent` with its `open_meteo_fetch` and `pagasa_fetch` tool runs, and so on). The weather and
  document agents also show their model calls with tokens and timings.
- **Privacy:** one shared client cleans every run before it is sent (`sanitize` in `utils/tracing.py`): phone-number-like digits become `[number]`,
  inputs named like a secret (key, token, password) become `[hidden]`, coordinates are rounded to 2 decimals (about 1 km) and long texts are cut.
  Message text is still sent, so do not trace real users without telling them.
- **Free plan:** one seat, 5,000 base traces a month, 14-day retention (reviewed 2026-10-29). One chat message is one trace. The SAM webcam
  and voice sockets are not traced, so they do not use the allowance.
- **Tests:** `backend/tests/conftest.py` switches tracing off so test runs never reach LangSmith.
- **Check it:** run a question through the graph and look for the `flood-gentic-chat` run in the LangSmith project, or list recent runs with the `langsmith` client.

## 7. Planned

- Railway services for the API and an ingestion worker, with the environment variables above set per service.
- Vercel frontend with `NEXT_PUBLIC_API_URL`, and the backend CORS list allowing its domains. The voice socket must be served over `wss`.
- A scheduled job that runs the full ingestion instead of by hand.
