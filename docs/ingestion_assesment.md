# Ingestion Assessment

What we measured on the real corpus before embedding it, and what we decided.
Design intent lives in `docs/ingestion_details.md`; this file records the evidence.
Numbers come from the notebooks in `notebooks/embeddings/` and the first full run (2026-10-28).

## 1. Corpus

| File | Type | Notes |
|---|---|---|
| `DISASTER PREPAREDNESS MANUAL.pdf` | Manual, 172 pages | Text layer; 16 pages have almost no text (images, blank pages) |
| `world-bank.pdf` | Report, 113 pages | Text layer; 1 near-empty page |
| `ncrpasig-marikina-laguna-de-bay.pdf` | Map sheet, 2 pages | Text layer, short |
| `20260914-NDRRMC-IRR-RA12287.pdf` | Law IRR, 17 pages | **Scanned: no text layer on any page** |
| `ph_emergency_contacts.csv` | Hotlines, 110 rows | National, city rescue, fire stations, hospitals |
| `ncr_locations.csv` | 17 sampled points | City, latitude, longitude, elevation (m) |

`ph_disaster_preparedness_manual.pdf` was a byte-identical copy of the manual and was removed.
The ingest script also skips documents with identical text.

## 2. PDF extraction (`notebooks/embeddings/ingestion_benchmarks.ipynb`)

| File | pypdf | PyMuPDF | Empty pages |
|---|---|---|---|
| Disaster manual (172 p) | 8.8 s | 0.9 s | 16 |
| World Bank (113 p) | 3.5 s | 0.2 s | 1 |
| Marikina / Laguna de Bay (2 p) | 0.17 s | 0.02 s | 0 |
| NDRRMC IRR (17 p) | 0.09 s | 0.02 s | **17** |

- Text quality is the same for both libraries (share of normal characters: 0.96 for the manual,
  0.99+ for the others). PyMuPDF is about 10x faster.
- The loader still uses `pypdf`. Speed is not a problem at this corpus size, so it was not switched.
- Before OCR the NDRRMC IRR produced **0 chunks**, so none of it was searchable.

## 3. Scanned PDF handling

| Option | Result | Decision |
|---|---|---|
| Tesseract OCR per page (`loaders.py`) | IRR yields 18 chunks, about 30 s for 17 pages | **Used.** Text goes through the normal chunker, BM25 and metadata filters |
| ColPali (`vidore/colSmol-500M`) page-image retrieval | recall@1 0.75, recall@3 0.75, MRR 0.79 on 8 questions; 11.8 s and 1.7 GB GPU for 17 pages | Not used in the pipeline. It finds pages but returns no text |

- OCR text has some noise from the letterhead logo and the "CERTIFIED TRUE COPY" stamp.
  Only English is installed for Tesseract (`eng`); the IRR is English.
- ColPali missed two questions (title of the IRR, definition of anticipatory actions). Kept as a
  fallback idea if OCR quality turns out to be a problem (`notebooks/embeddings/colpali_scanned_pdf.ipynb`).
- Its labels cover only pages 1 and 5, so recall is a lower bound.

## 4. Embedding model (`notebooks/embeddings/embedding_readiness.ipynb`)

214 chunks from the manual, World Bank report and Marikina sheet (the scanned IRR had no text yet), 9 scored questions (English and Filipino).

| Model | Dim | Embed time | Peak GPU | recall@5 | MRR | MRR EN | MRR TL |
|---|---|---|---|---|---|---|---|
| bge-m3 | 1024 | 26.3 s | 3.1 GB | 0.78 | 0.66 | 0.75 | 0.55 |
| **multilingual-e5-large** | 1024 | 17.3 s | 2.7 GB | 1.00 | 0.79 | 0.85 | 0.71 |
| multilingual-e5-base | 768 | 5.1 s | 2.2 GB | 0.89 | 0.75 | 0.84 | 0.64 |

**Decision:** `intfloat/multilingual-e5-large` is the default (1024 dimensions, so the pgvector
column does not change). It has the best score, is faster than bge-m3, and is better on Filipino.

Caveats:
- The question set is small and relevance is decided by keywords, so only the bge-m3 vs e5 gap is
  large enough to trust. Re-run on the real golden set (`docs/evaluation_rubric.md`) before tuning.
- E5 needs `query: ` / `passage: ` prefixes. `embeddings.py` adds them for any model with "e5"
  in its name.
- Changing the model means re-embedding everything.

## 5. First full run

359 chunks, embedded with multilingual-e5-large and saved to
`faiss_vectors/20261028-195042_multilingual-e5-large/`.

| Source | Chunks |
|---|---|
| Disaster manual | 97 |
| World Bank report | 115 |
| NDRRMC IRR (OCR) | 18 |
| Marikina / Laguna de Bay | 2 |
| Emergency contacts | 110 |
| NCR locations | 17 |

Spot checks on the saved index:

| Question | Top result |
|---|---|
| Who do I call for a flood rescue in Marikina? | Marikina City Rescue numbers |
| Sino ang puwedeng mag-lift ng state of imminent disaster? | NDRRMC IRR sections (the OCR'd scan) |
| What should go in an emergency go-bag? | Manual supply list ranked 2nd; a health-station checklist table ranked 1st |

## 6. Known issues and next steps

1. **Supabase write not done.** Connection is ready (`SUPABASE_PROJECT_REF`, `SUPABASE_PASSWORD`, `SUPABASE_POOLER_HOST` in `backend/.env`,
   or `SUPABASE_DB_URL`) but the database has no `vector` extension or `chunks` table yet; then run the ingest
   script with `--supabase`. Until then retrieval only exists as FAISS files.
2. **Table-like pages in the manual** (checklists, blank forms) become low-value chunks that can
   outrank real content. Try dropping chunks with very little prose, or rely on the reranker.
3. **`ncr_locations.csv`** is numbers (elevation, coordinates). Embeddings handle these poorly and
   only 7 distinct coordinates exist. A structured lookup by city is the better use; the chunks are
   kept for now so questions about a city's elevation can surface.
4. **No golden set yet.** All quality numbers above are from a starter question set.
5. **BM25 index not built yet** (`bm25.py` exists but is not part of the saved run).
6. **No Tagalog OCR or text.** The manual and reports are English; Filipino only appears in queries.
