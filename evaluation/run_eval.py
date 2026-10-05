"""Lightweight evaluation: runs fixed questions through the pipeline and
saves answers + per-stage latency.   Run from the project root:

    python -m evaluation.run_eval
"""

import pandas as pd

from src.config import project_path
from src.pipeline import DriveWise

EVAL_QUESTIONS = [
    "What is the mileage or fuel efficiency of this car?",
    "How many standard safety features does the car have?",
    "What is the engine or motor's power output?",
    "What is the seating capacity?",
    "What are the exterior dimensions of the car?",
    "What infotainment features are available?",
    "What is the warranty offered on this vehicle?",
    "What variants or trims are available?",
    # Out-of-scope question: a grounded assistant should say the brochure doesn't mention it
    "Who won the 2018 FIFA World Cup?",
]


def main() -> None:
    assistant = DriveWise()
    brand = assistant.store.brands()[0]
    model = assistant.store.models(brand)[0]
    print(f"Evaluating on: {brand} {model}\n")

    records = []
    for question in EVAL_QUESTIONS:
        r = assistant.ask(question, brand, model)
        records.append(
            {
                "question": question,
                "status": r.status,
                "retrieval_s": round(r.retrieval_time, 4),
                "rerank_s": round(r.rerank_time, 4),
                "generation_s": round(r.generation_time, 4),
                "total_s": round(r.retrieval_time + r.rerank_time + r.generation_time, 4),
                "sources": len(r.sources),
                "answer": r.answer,
            }
        )
        print(f"[{r.status}] {question}")

    df = pd.DataFrame(records)
    out_dir = project_path(assistant.cfg["paths"]["evaluation"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "eval_results.csv", index=False)

    print("\n✓ Evaluation summary")
    print(f"  Questions: {len(df)}")
    print(f"  Success rate: {(df['status'] == 'success').mean() * 100:.1f}%")
    print(f"  Avg total latency: {df['total_s'].mean():.3f}s")
    print(f"  Saved to: {out_dir / 'eval_results.csv'}")


if __name__ == "__main__":
    main()
