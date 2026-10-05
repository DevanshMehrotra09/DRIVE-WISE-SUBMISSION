"""End-to-end RAG pipeline: filter -> retrieve -> re-rank -> prompt -> Gemini -> log."""

import json
import time
from dataclasses import dataclass, field

from src.config import load_config, load_prompts, project_path
from src.generator import Generator, build_prompt
from src.retriever import Retriever, VectorStore


@dataclass
class RagResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0
    status: str = "success"


class DriveWise:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.prompts = load_prompts(self.cfg)
        self.store = VectorStore(project_path(self.cfg["paths"]["vectorstore"]))
        self.retriever = Retriever(self.cfg, self.store)
        self.generator = Generator(self.cfg, self.prompts)
        self.log_file = project_path(self.cfg["paths"]["logs"]) / "query_log.jsonl"

    def ask(self, question: str, brand: str, model: str) -> RagResponse:
        top_k = self.cfg["retrieval"]["top_k"]
        top_n = self.cfg["retrieval"]["top_n"]

        candidates, retrieval_time = self.retriever.retrieve(question, brand, model, top_k)
        if candidates.empty:
            response = RagResponse(
                answer=f"No brochure data found for {brand} {model}.",
                retrieval_time=retrieval_time,
                status="no_context",
            )
            self._log(question, brand, model, response)
            return response

        context, rerank_time = self.retriever.rerank(question, candidates, top_n)
        prompt = build_prompt(self.prompts, question, brand, model, context)

        start = time.perf_counter()
        answer, status = self.generator.generate(prompt)
        generation_time = time.perf_counter() - start

        response = RagResponse(
            answer=answer,
            sources=[
                {"document": r.document_name, "page": int(r.page), "section": r.section}
                for r in context.itertuples()
            ],
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            status=status,
        )
        self._log(question, brand, model, response)
        return response

    def _log(self, question: str, brand: str, model: str, r: RagResponse) -> None:
        """Append one JSON line per query, including per-stage latency."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": question,
            "brand": brand,
            "model": model,
            "response_time_s": round(r.retrieval_time + r.rerank_time + r.generation_time, 4),
            "retrieval_latency_s": round(r.retrieval_time, 4),
            "rerank_latency_s": round(r.rerank_time, 4),
            "generation_latency_s": round(r.generation_time, 4),
            "retrieved_chunks": len(r.sources),
            "sources": r.sources,
            "status": r.status,
        }
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"⚠ Could not write log: {exc}")
