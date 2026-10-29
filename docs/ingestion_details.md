# Ingestion and Retrieval Details

How Flood-Gentic chunks, indexes, retrieves, and reranks disaster information.
Every number in this document (chunk sizes, `k` values, thresholds) is a **starting value**
to be tuned against the evaluation golden set described in section 8.

## 1. Corpus and why it shapes the design

| Source type | Examples | Traits |
|---|---|---|
| Official advisories | PAGASA bulletins, NDRRMC situation reports | Short, time-sensitive, expire quickly |
| Structured lists | Evacuation centers, hotlines | Row-like records, exact names and numbers |
| Long guides | NDRRMC / OCD preparedness manuals | Long, hierarchical, rarely change |
| Community reports | Resident incident and supply-request reports | Noisy, unverified, short |
| Weather data | Open-Meteo forecasts | Structured, refreshed often |
| Scanned documents | NDRRMC IRR of RA 12287 | No text layer; must be OCR'd before chunking |
| Location data | NCR points with elevation | Numbers, not prose; weak for embeddings |

Files currently ingested from `corpus/` are listed with measurements in `docs/ingestion_assesment.md`.

Queries mix English and Filipino and often contain exact tokens: typhoon names ("Carina"),
city and barangay names, "Signal No. 3", phone numbers. Dense embeddings handle paraphrase and
language mixing well but are weak on exact tokens. BM25 is the opposite. That is why retrieval
is hybrid. Freshness also matters: an expired advisory is worse than no advisory.

## 2. Chunking (per source type)

One splitter for everything would cut bulletins mid-warning and split evacuation records in
half, so the strategy depends on the source.

| Source type | Strategy | Size / overlap |
|---|---|---|
| Advisories, bulletins | Structure-aware split on headings and sections. Keep whole if under ~600 tokens. | 300-500 tokens, ~10% overlap |
| Manuals, guides | Heading-aware recursive split (Markdown or PDF section hierarchy). Prepend the heading path to each chunk. | 400-600 tokens, 50-80 overlap |
| Evacuation centers, hotlines | No splitting. One record per row, rendered as a sentence: "Center X, Barangay Y, City Z, capacity N." Hotlines: "Marikina City Rescue (Local - City Rescue), Landline: (02) 8646-2426." | One record = one chunk |
| NCR location points | One record per point, grouped by city: "Manila location 2: elevation 9 m, latitude ..., longitude ...". City stored in metadata. | One point = one chunk |
| Community reports | One chunk per report. Trust level kept in metadata. | One report = one chunk |

Sizes in code are counted in **words** (about 1.3 tokens per word): manuals use 400 words with
50 overlap, advisories 300 with 30. PDF text is read page by page; a page with fewer than 50
characters is treated as a scan and read with Tesseract OCR (`loaders.py`).

### Chunking decision flow

```mermaid
flowchart TD
    A[Incoming document] --> B{Source type?}
    B -->|Advisory / bulletin| C{Under ~600 tokens?}
    C -->|Yes| C1[Keep whole bulletin as 1 chunk]
    C -->|No| C2[Split by section<br/>300-500 tokens, ~10% overlap]
    B -->|Manual / guide| D[Heading-aware recursive split<br/>400-600 tokens, 50-80 overlap<br/>prepend heading path]
    B -->|Evacuation center / hotline| E[1 row = 1 chunk<br/>render as a sentence]
    B -->|Community report| F[1 report = 1 chunk<br/>store trust level]
    C1 --> G[Add context header + metadata]
    C2 --> G
    D --> G
    E --> G
    F --> G
    G --> H{content_hash<br/>already stored?}
    H -->|Yes| I[Skip]
    H -->|No| J[Upsert by doc_id + chunk_idx]
```

### Contextual enrichment

Prepend a one-line header to every chunk before embedding **and** before BM25 indexing:

```
[PAGASA | Tropical Cyclone Bulletin | 2026-10-24 | Metro Manila | typhoon] <chunk text>
```

This keeps a chunk like "Signal No. 3 is raised over..." understandable on its own and gives
BM25 the agency, hazard, and area terms to match.

### Metadata stored per chunk

`doc_id`, `chunk_idx`, `source`, `agency`, `doc_type`, `hazard`, `city`, `barangay`,
`issued_at`, `valid_until`, `language`, `trust_tier`, `content_hash`.

### Idempotent re-ingestion

Dedupe on `content_hash`. Upserts are keyed on `doc_id` + `chunk_idx`, so re-running an
ingestion job never duplicates chunks.

## 3. Hybrid retrieval

### Indexing path (offline)

```mermaid
flowchart LR
    S1[PAGASA / NDRRMC] --> F
    S2[Open-Meteo] --> F
    S3[Evacuation lists] --> F
    S4[Community reports] --> F
    F[Fetch] --> N[Clean / normalize]
    N --> CH[Chunk by source type]
    CH --> EN[Enrich:<br/>context header + metadata]
    EN --> EM[Embed<br/>multilingual-e5-large]
    EN --> BI[Tokenize]
    EM --> PG[(Supabase pgvector<br/>HNSW, cosine)]
    EM --> FX[(FAISS run folder<br/>faiss_vectors/)]
    BI --> BM[(BM25 index<br/>bm25s, in process)]
```

### Query path (online)

```mermaid
flowchart TD
    Q[User question] --> R{Short or<br/>conversational?}
    R -->|Yes, no exact IDs| RW[LLM rewrite +<br/>Filipino-English expansion]
    R -->|No| FI
    RW --> FI[Metadata filter:<br/>hazard, area, valid_until >= now]
    FI --> D[Dense search<br/>top 30]
    FI --> B[BM25 search<br/>top 30]
    D --> RRF[RRF fusion, k=60<br/>~40 candidates]
    B --> RRF
    RRF --> RK[Cross-encoder rerank<br/>top 5-8]
    RK --> BR[Business rules:<br/>expiry, recency, trust, diversity cap]
    BR --> T{Any chunk above<br/>score threshold?}
    T -->|Yes| LLM[LLM answer with citations]
    T -->|No| FB[No verified info +<br/>official hotlines]
```

### Request sequence

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI / LangGraph
    participant PG as Supabase pgvector
    participant BM as BM25 index
    participant RR as Reranker
    participant L as LLM

    U->>API: Question
    API->>API: Detect exact IDs, set RRF weights
    par Dense
        API->>PG: Vector search + filter (top 30)
        PG-->>API: Ranked chunks
    and Sparse
        API->>BM: BM25 query + filter (top 30)
        BM-->>API: Ranked chunks
    end
    API->>API: RRF fuse to ~40
    API->>RR: Query + candidates
    RR-->>API: Scored top 5-8
    API->>API: Apply expiry / trust / diversity rules
    API->>L: Prompt with cited chunks
    L-->>U: Grounded answer
```

### Dense retriever
- Multilingual embedding model: **`intfloat/multilingual-e5-large`** (1024 dimensions, Tagalog
  support is required). Chosen over `bge-m3` and `multilingual-e5-base` in `docs/ingestion_assesment.md`.
  Override with the `EMBEDDING_MODEL` environment variable; changing it means re-embedding everything.
- E5 models need `query: ` / `passage: ` in front of the text. `embeddings.py` adds them
  (chunks as passages, questions as queries).
- Stored in Supabase pgvector with an HNSW index, cosine distance.
- Also saved as a local FAISS snapshot (see "FAISS snapshots" below).

### FAISS snapshots

`scripts/ingest_corpus.py` writes every run to its own folder, so earlier runs are never overwritten:

```
faiss_vectors/<YYYYmmdd-HHMMSS>_<model name>/
    index.faiss     vectors, exact cosine search (IndexFlatIP on normalized vectors)
    chunks.json     id, doc_id, chunk_idx, text and metadata; row i matches index row i
    metadata.json   model, vector size, time, chunk count, SHA-256 of each source file
```

Code: `backend/src/rag/faiss_store.py` (`save_run`, `load_run`). Supabase is written only with
`--supabase` and needs a Supabase connection in `backend/.env`: `SUPABASE_DB_URL`, or
`SUPABASE_PROJECT_REF` + `SUPABASE_PASSWORD` + `SUPABASE_POOLER_HOST` (see `backend/.env.example`).

### Sparse retriever (BM25)
- Real BM25, not Postgres `ts_rank`, which has no inverse-document-frequency saturation.
- Supabase does not offer `pg_search`, and the corpus is small (thousands of chunks), so
  recommend `bm25s` (or `rank_bm25`) built in process and rebuilt after each ingestion run.
- Tokenizer must work for English and Filipino: lowercase, no stemming that mangles Tagalog
  affixes, and keep numbers, phone numbers, and proper nouns as single tokens.
- Upgrade path if the corpus grows past roughly 1M chunks or several backend replicas need a
  shared index: ParadeDB (`pg_search`) or OpenSearch.

### Metadata pre-filter
Apply the same filter to both retrievers: hazard type, city/area, and `valid_until >= now`.
Filtering before ranking keeps expired or out-of-area chunks out of the candidate set.

### Fusion: Reciprocal Rank Fusion
- `score(d) = sum over retrievers of weight / (60 + rank(d))`
- Worked example (weights 1.0 / 1.0): chunk A is rank 1 in dense and rank 5 in BM25, chunk B is
  rank 2 in dense only.

```mermaid
flowchart LR
    subgraph Dense
        d1[1. A]
        d2[2. B]
    end
    subgraph BM25
        b1[1. C]
        b5[5. A]
    end
    d1 --> sA["A = 1/61 + 1/65 = 0.0318"]
    b5 --> sA
    d2 --> sB["B = 1/62 = 0.0161"]
    b1 --> sC["C = 1/61 = 0.0164"]
    sA --> o["Fused order: A, C, B"]
    sC --> o
    sB --> o
```

  A chunk found by both retrievers beats chunks found by only one, without comparing raw scores.
- Take top 30 from each retriever, fuse to about 40 candidates.
- RRF uses ranks only, so no score normalization between BM25 and cosine is needed.
- Default weights 1.0 / 1.0. Raise the BM25 weight when the query contains an exact identifier
  (typhoon name, signal number, phone number, place name).

### Optional query step
LLM query rewrite and Filipino-to-English expansion, only when the query is conversational or
very short. Skip it for queries that already contain exact identifiers.

## 4. Reranking

Candidate funnel: each stage narrows the set and costs more per chunk than the one before.

```mermaid
flowchart TD
    A["Whole corpus<br/>thousands of chunks"] -->|metadata filter| B["Filtered pool"]
    B -->|dense top 30 + BM25 top 30| C["~40 fused candidates<br/>(cheap: vector + keyword)"]
    C -->|cross-encoder<br/>(expensive: reads query + chunk)| D["Top 5-8"]
    D -->|expiry, trust, diversity, threshold| E["Final context for LLM"]
```

1. **Cross-encoder rerank** of the ~40 fused candidates down to the top 5-8.
   - First choice: `bge-reranker-v2-m3` (multilingual, self-hosted, no per-call cost).
   - Hosted alternative: Cohere Rerank 3.5 multilingual, if CPU latency on Railway is a problem.
2. **Business rules after reranking**
   - Drop expired advisories; boost recent ones.
   - Boost official `trust_tier` sources over community reports.
   - Drop anything below a minimum rerank score.
3. **Diversity**: cap at 2-3 chunks per `doc_id` (or use MMR) so one bulletin cannot fill the context.
4. **Safety fallback**: if nothing clears the threshold, the assistant says it has no verified
   information and points to official hotlines. It must not improvise safety guidance.
5. **Prompt assembly**: each chunk is passed with its source and issued time so the answer can cite them.

## 5. Freshness and scheduling

| Data | Refresh | Expiry |
|---|---|---|
| Open-Meteo forecasts | Every 15-60 min | Short `valid_until` |
| Advisories / bulletins | Every 15-60 min | `valid_until` from the bulletin, else issue time + 24 h |
| Community reports | On submit | Review window, then downgrade |
| Manuals, evacuation lists | On change | None (re-ingest on update) |

Expired chunks are soft-deleted (filtered by `valid_until`) rather than physically removed, so
past answers stay auditable.

```mermaid
stateDiagram-v2
    [*] --> Ingested: upsert chunk
    Ingested --> Active: embedded + BM25 indexed
    Active --> Superseded: newer bulletin replaces it
    Active --> Expired: valid_until < now
    Superseded --> Archived
    Expired --> Archived: excluded from retrieval,<br/>kept for audit
    Archived --> [*]
```

## 6. LangChain / LangGraph mapping

Code lives in `backend/src/rag/`: `loaders.py` -> `chunking.py` -> `embeddings.py` -> `vector_store.py` (write path);
`retriever.py` (`HybridRetriever`: `vector_store` + `bm25.py` + `fusion.py`) -> `reranking.py` -> `rules.py` (read path).

- Retrieval is a graph node: filter -> dense + BM25 -> RRF -> rerank -> rules.

```mermaid
flowchart LR
    START([start]) --> route[route_query]
    route --> rewrite[rewrite_query]
    route --> retrieve
    rewrite --> retrieve[hybrid_retrieve<br/>dense + BM25 + RRF]
    retrieve --> rerank[rerank + rules]
    rerank --> check{enough evidence?}
    check -->|yes| answer[generate_answer]
    check -->|no| fallback[hotline_fallback]
    answer --> END([end])
    fallback --> END
```

- Fusion: `EnsembleRetriever` (RRF built in) or a small custom RRF function if per-query weights are needed.
- Reranking: `ContextualCompressionRetriever` wrapping the cross-encoder.
- Already in `backend/pyproject.toml`: `langchain`, `langchain-community`, `langgraph`.
- Added so far: `sentence-transformers`, `pypdf`, `pymupdf`, `faiss-gpu`, `psycopg[binary]`,
  `colpali-engine` (benchmark only), `torch`. The Tesseract program must be installed for scans.
- Still to add: `bm25s`. (`pgvector` is not needed: embeddings are sent as text and cast with `::vector`.)

## 7. Phased rollout

| Phase | Scope |
|---|---|
| 1 (MVP) | Per-source chunking, enrichment, dense + BM25, RRF |
| 2 | Cross-encoder reranker, score threshold, safety fallback |
| 3 | Query rewrite, recency / trust rules, diversity cap |

```mermaid
flowchart LR
    P1["Phase 1 (MVP)<br/>chunking, enrichment<br/>dense + BM25 + RRF"] --> P2["Phase 2<br/>cross-encoder<br/>threshold + fallback"]
    P2 --> P3["Phase 3<br/>query rewrite<br/>recency / trust rules"]
    P1 -.->|gate: golden set baseline| P2
    P2 -.->|gate: rerank beats hybrid| P3
```

## 8. Evaluation

Tie into `docs/evaluation_rubric.md`.

- Build a golden set of about 50 bilingual (English / Filipino) NCR queries with labeled relevant chunks.
  So far only a 9-question keyword-labelled starter set exists (`notebooks/embeddings/embedding_readiness.ipynb`).
- Metrics: recall@k, MRR, nDCG@5, and answer groundedness (does the answer cite retrieved text).
- Ablation to justify each stage: dense only, BM25 only, hybrid, hybrid + rerank.
- Tune chunk sizes, `k`, RRF weights, and the rerank threshold on this set.

## 9. Open risks

- **Filipino tokenization**: no mature BM25 tokenizer; validate against the golden set.
- **Reranker latency**: roughly 200-500 ms for 40 candidates on CPU. Mitigate by trimming to 20-30
  candidates or using a hosted reranker.
- **In-process BM25 index**: rebuilt on ingest and duplicated per replica. Fine for MVP; revisit with ParadeDB / OpenSearch if it grows.
- **Scanned documents**: OCR text has noise (logos, stamps) and is English only. Check OCR'd
  chunks by eye before trusting citations from them.
- **Table-like manual pages**: checklists and blank forms make low-value chunks that can rank high.
- **Numeric location data**: embeddings are weak on coordinates and elevations; prefer a
  structured lookup by city for flood-risk logic.
- **Community report quality**: unverified content must never outrank official sources on safety-critical questions.
