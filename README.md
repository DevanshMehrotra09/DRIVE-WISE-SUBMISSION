# DriveWise — Metadata-Aware Automotive RAG Assistant

DriveWise answers natural-language questions about a specific car (brand + model) using
Retrieval-Augmented Generation. Every answer is produced by the **Google Gemini API**
strictly from retrieved brochure excerpts, with the document, section and page cited.

## How the LLM is used

1. The user picks a brand/model and asks a question.
2. Only that vehicle's chunks are searched (metadata filter), then re-ranked by a cross-encoder.
3. The top chunks are inserted into the prompt defined in `prompts/prompts.yaml`.
4. Gemini (`generation` and `models.llm` settings in `config.yaml`) writes a grounded, cited answer.
   The rules live in the system instruction; the user turn carries only the vehicle, excerpts and question.
5. Rate-limit errors are retried with exponential backoff; every query is logged to `logs/query_log.jsonl`.

## Project structure

```
DriveWise/
├── config.yaml               # models, retrieval/generation settings, paths, section keywords
├── prompts/prompts.yaml      # PROMPT FILE: system instruction + user template
├── src/
│   ├── config.py             # loads config.yaml and the prompt file
│   ├── ingest.py             # PDFs -> chunks -> embeddings -> FAISS index  (python -m src.ingest)
│   ├── retriever.py          # metadata filtering, vector search, cross-encoder re-ranking
│   ├── generator.py          # prompt building + Gemini API call with retry
│   └── pipeline.py           # end-to-end pipeline + JSONL logging
├── app/app.py                # Streamlit demo
├── evaluation/run_eval.py    # latency / success evaluation  (python -m evaluation.run_eval)
├── Data/Brochures/<BRAND>/<Model>.pdf
├── vectorstore/              # persisted FAISS index + chunk metadata
├── .env.example              # copy to .env and add GOOGLE_API_KEY
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env  -> then add your GOOGLE_API_KEY
python -m src.ingest          # only needed if vectorstore/ is missing or brochures changed
streamlit run app/app.py
```

Run all commands from the project root. Python 3.10+.

## Configuration

Everything tunable is in `config.yaml`: Gemini model name, temperature, max output tokens,
retry settings, `top_k` / `top_n`, embedding and re-ranker models, folder paths and the
section-classification keywords. To change the model, edit `models.llm` only.

## Evaluation

```bash
python -m evaluation.run_eval
```

Runs 9 questions (including one out-of-scope question to check the model refuses to answer
from outside knowledge) and writes answers plus per-stage latency to `evaluation/eval_results.csv`.

## Limitations

- Section classification is keyword-frequency based, not a learned classifier.
- Chunking is page-level; long multi-topic pages are not split further.
- Evaluation measures latency and success rate, not answer correctness.

## Future improvements

- Section-aware chunking for dense pages.
- Ground-truth correctness / faithfulness evaluation (e.g. Ragas).
- Per-brochure embedding cache to avoid full re-ingestion.
