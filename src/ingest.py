"""Offline ingestion: brochure PDFs -> page chunks -> embeddings -> FAISS index.

Run from the project root:
    python -m src.ingest
"""

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import load_config, project_path


@dataclass
class Brochure:
    brand: str
    model: str
    path: Path

    @property
    def document_name(self) -> str:
        return self.path.name


@dataclass
class Chunk:
    chunk_id: str
    brand: str
    model: str
    document_name: str
    document_version: str
    page: int
    section: str
    text: str


def discover_brochures(brochure_dir: Path) -> list[Brochure]:
    """Find brochures stored as <brochure_dir>/<Brand>/<Model>.pdf."""
    if not brochure_dir.exists():
        raise FileNotFoundError(f"Brochure directory not found: {brochure_dir}")

    brochures = [
        Brochure(brand=brand_dir.name, model=pdf.stem, path=pdf)
        for brand_dir in sorted(brochure_dir.iterdir())
        if brand_dir.is_dir()
        for pdf in sorted(brand_dir.glob("*.pdf"))
    ]
    if not brochures:
        raise RuntimeError(f"No brochure PDFs found in {brochure_dir}")
    return brochures


def classify_section(text: str, section_keywords: dict[str, list[str]]) -> str:
    """Keyword-frequency section label (deterministic, free, fast)."""
    lower = text.lower()
    scores = {
        section: sum(lower.count(kw) for kw in keywords)
        for section, keywords in section_keywords.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def make_chunk_id(brand: str, model: str, page: int) -> str:
    return hashlib.md5(f"{brand}|{model}|{page}".encode()).hexdigest()[:12]


def build_chunks(
    brochures: list[Brochure],
    section_keywords: dict[str, list[str]],
    version: str = "v1",
) -> list[Chunk]:
    """One chunk per non-empty page, tagged with brand/model/page/section."""
    chunks = []
    for brochure in brochures:
        try:
            with fitz.open(brochure.path) as doc:
                for page_number, page in enumerate(doc, start=1):
                    text = page.get_text().strip()
                    if not text:
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(brochure.brand, brochure.model, page_number),
                            brand=brochure.brand,
                            model=brochure.model,
                            document_name=brochure.document_name,
                            document_version=version,
                            page=page_number,
                            section=classify_section(text, section_keywords),
                            text=text,
                        )
                    )
        except Exception as exc:
            print(f"✗ Skipping {brochure.document_name}: {exc}")
    if not chunks:
        raise RuntimeError("No text could be extracted from the brochures.")
    return chunks


def build_index(chunks: list[Chunk], embedder: SentenceTransformer) -> faiss.Index:
    """Embed chunks and store them in an exact inner-product (cosine) index."""
    embeddings = embedder.encode(
        [c.text for c in chunks], convert_to_numpy=True, show_progress_bar=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_vector_store(index: faiss.Index, chunks: list[Chunk], directory: Path) -> None:
    """Persist the index and its row-aligned chunk metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "index.faiss"))
    pd.DataFrame([asdict(c) for c in chunks]).to_csv(
        directory / "chunk_metadata.csv", index=False
    )


def main() -> None:
    cfg = load_config()
    start = time.perf_counter()

    brochures = discover_brochures(project_path(cfg["paths"]["brochures"]))
    print(f"✓ Found {len(brochures)} brochures")

    chunks = build_chunks(brochures, cfg["section_keywords"])
    print(f"✓ Built {len(chunks)} chunks")

    embedder = SentenceTransformer(cfg["models"]["embedding"])
    index = build_index(chunks, embedder)

    out_dir = project_path(cfg["paths"]["vectorstore"])
    save_vector_store(index, chunks, out_dir)
    print(f"✓ Saved {index.ntotal} vectors to {out_dir} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
