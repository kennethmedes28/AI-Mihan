# Agent Duration Benchmarks

How long each Flood-Gentic chat agent takes to answer, and how two Muse Spark models compare.
Agent design is in `docs/backend_architecture.md`; retrieval design is in `docs/ingestion_details.md`.
Run date: 2026-10-28. Every number is a median of a small number of runs on one laptop
(RTX 4050 6 GB, WSL2), so treat differences under about 1 second as noise.

Notebooks (in `notebooks/agent/`):

| Notebook | Model |
|---|---|
| `muse_spark_1.2_contributor_agent_duration.ipynb` | `muse-spark-1.2-contributor` |
| `muse_spark_1.3_agent_duration.ipynb` | `muse-spark-1.3` (standard) |

Both are the same notebook with a different `MODEL` value.

## 1. Method

Each agent is called the way production calls it: one question through the compiled chat graph
(`langgraph_orchestrator/graph.py`), which includes the keyword router. Per question the notebook records:

- **total**: wall-clock time of the graph call,
- **model**: time inside Muse calls, measured with a LangChain callback (also counts output tokens),
- **other**: total minus model (downloads, retrieval, parsing).

Each question runs **5 cold runs** (all caches cleared first: forecast, PAGASA pages, river data,
geocoding, answers, query embeddings, hourly model-call cap) plus **1 warm run** to show what caching saves.
Model calls use `max_retries=0`, so a failure is counted, not hidden by a retry. A run counts as a
"fallback" if the reply is one of the fixed error texts; none occurred in either benchmark.

### What is real and what is stubbed

| Part | In the benchmark |
|---|---|
| Muse model calls (weather, documents) | Real |
| Open-Meteo, PAGASA pages, Nominatim geocoding | Real, over the network |
| Document search | Real FAISS run (`faiss_vectors/`) and the real embedding model, **dense only**: no BM25, no database |
| Supabase reads and writes | **Stubbed**: three synthetic evacuation centers per city, no community reports, report writes ignored. Database latency is not included |
| OpenRouteService routing | **Not available** (no `ORS_API_KEY`): the route agent times only its fallback path |
| Cross-encoder reranker | Off for the main table (`RERANKER_MODEL=none`); measured separately in section 4 |

### Questions

| Agent | Question |
|---|---|
| emergency | Tulong! Nalulunod kami sa baha sa Marikina, may bata kaming kasama |
| report | Report: flooded na ang kalsada sa Barangay Malanday, Marikina, hanggang tuhod ang tubig |
| route | How do I get to the nearest evacuation center from Marikina? |
| evacuation | Where is the nearest evacuation center in Pasig? |
| contacts | What is the hotline number for Quezon City? |
| location | Marikina |
| flood_risk | Will Marikina flood today? |
| weather | Will it rain tomorrow in Quezon City? |
| documents | How do I prepare my family before a typhoon? |

The notebook checks with `classify_intent` that each question reaches the intended agent.

## 2. Results by agent

Median cold time over 5 runs. Model rows are the agents that call Muse; the others make no model call.

| Agent | 1.2 Contributor | 1.3 | Notes |
|---|---|---|---|
| contacts | 1 ms | 1 ms | CSV lookup |
| location | 1 ms | 1 ms | alias match |
| emergency | 2 ms | 2 ms | fixed text |
| report | 69 ms | 91 ms | database write stubbed |
| route | 1.96 s | 2.32 s | fallback path only (no routing key); mostly the Open-Meteo risk check |
| flood_risk | 5.10 s | 3.95 s | Open-Meteo downloads |
| evacuation | 11.43 s | 2.18 s | see "Network noise" below |
| weather | 7.84 s | 7.13 s | model 6.57 s vs 5.76 s |
| documents | 7.91 s | 7.99 s | model 7.65 s vs 7.68 s |

Warm repeats (caches full) were about 1 ms for every agent, except evacuation (1.5 s) and route (1.0 s) on the
1.2 run.

The router itself (`classify_intent`) takes about 0.003 ms per call.

### Network noise

Open-Meteo requests sometimes failed with `_ssl handshake timed out`. The code waits up to 10 seconds per
request and counts that wait as "other" time. This is why evacuation shows 11.4 s on 1.2 Contributor (three of its
runs hit timeouts; its fastest run was 1.9 s) and why route, flood_risk and evacuation show maximums of 11 to 13 s
on both runs. These three agents make no model call, so their differences between the two benchmarks are
network variation, not model differences.

## 3. Muse Spark 1.2 Contributor vs 1.3

Only the weather and documents agents call the model, so only they are a fair comparison.

| Agent | Measure | 1.2 Contributor | 1.3 |
|---|---|---|---|
| weather | total (range) | 7.84 s (6.3-9.8) | 7.13 s (6.8-9.3) |
| weather | model time / output tokens | 6.57 s / 1,498 | 5.76 s / 1,454 |
| documents | total (range) | 7.91 s (5.8-9.0) | 7.99 s (7.7-9.3) |
| documents | model time / output tokens | 7.65 s / 1,855 | 7.68 s / 2,332 |

- At default settings latency is about the same; the ranges overlap.
- 1.3 produced tokens somewhat faster: about 253 and 304 tokens per second versus 228 and 243 (derived from the
  table above). On the documents question it wrote about 26% more output in the same time.
- Quality was **not** compared. All 45 runs per model returned real answers, but no one graded them.
- Runs happened at different times, so API load may differ between them.

### Reasoning effort matters more than the model

Muse is a reasoning model, so much of its time is hidden thinking that is billed as output.
`reasoning_effort="minimal"` cut output tokens by roughly 4 to 5 times (3 runs per cell, so noisy):

| Agent | Model | default total | minimal total | default tokens | minimal tokens |
|---|---|---|---|---|---|
| documents | 1.2 Contributor | 7.09 s | 3.16 s | 2,009 | 510 |
| documents | 1.3 | 8.99 s | 2.14 s | 2,674 | 506 |
| weather | 1.2 Contributor | 13.25 s | 3.72 s | 1,471 | 362 |
| weather | 1.3 | 6.19 s | 3.35 s | 1,519 | 324 |

The 13.25 s weather figure for 1.2 Contributor is well above its 7.84 s main-run median; that sample included
an Open-Meteo failure, so do not rely on it.

## 4. Cross-encoder reranker

The real documents agent reranks candidates with `BAAI/bge-reranker-v2-m3`. With it on:

| Model | total (median) | model | other |
|---|---|---|---|
| 1.2 Contributor | 29.05 s | 9.09 s | 18.48 s |
| 1.3 | 28.12 s | 8.83 s | 18.68 s |

The reranker adds about 18.5 s on either model, more than the model call itself. It was not checked whether the
reranker ran on the GPU, so the cause is unconfirmed. Loading it took 190 s the first time (including the download)
and 45 s the second time; that is a one-off cost, not per question.

## 5. Other findings

- **Cold starts are not in the tables.** First use of the embedding model took about 90 s (92 s and 88 s).
  Load models at application start, not on the first request.
- **Routing issue:** "What should go in an emergency go-bag?" is classified as an emergency (the word "emergency"
  matches first), so it gets the fixed emergency reply instead of the documents answer. The benchmark uses a
  different documents question because of this. The router is `langgraph_orchestrator/intents.py`.
- **Route agent:** its real routing path (OpenRouteService) was never timed.

## 6. Recommendations

1. For chat latency, set `reasoning_effort="minimal"` on the weather and documents agents; it is the largest
   lever (about 2 to 4 times faster in these runs). Check answer quality at that setting before adopting it.
2. Choose between 1.2 Contributor and 1.3 on answer quality and per-token cost, not latency; latency is
   about equal here.
3. Investigate the reranker (device, number of candidates, chunk length) before enabling it in production.
4. Fix the "emergency" keyword misrouting for preparedness questions.
5. Add an Open-Meteo timeout shorter than 10 s with one retry, or serve the last good forecast; a single
   stalled request dominates the data agents' worst case.

## 7. Limits and next steps

- Small samples (5 runs, 3 for the effort comparison), one machine, different times of day.
- Re-run with a real Supabase connection and an `ORS_API_KEY` to time the evacuation, report and route agents fully.
- Add BM25 to the benchmark retriever so the documents agent matches production.
- Save run results to CSV so the two models can be compared in a notebook instead of by reading outputs.
- Grade a sample of answers from both models using the golden set in `docs/evaluation_rubric.md`.
