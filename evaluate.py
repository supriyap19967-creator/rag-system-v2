import csv
import os
import uuid
from pathlib import Path

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from langchain_openai import ChatOpenAI

from app.embeddings import get_bge_embeddings


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/query")
OUTPUT_CSV = Path("evaluation/ragas_scores.csv")

QUESTIONS = [
    "What was India's GDP in 2022?",
    "What were India's CO2 emissions per capita in 2022?",
    "What was the GDP of the United States in 2022?",
    "What were China's CO2 emissions per capita in 2022?",
    "What was Brazil's GDP in 2021?",
]


def run_rag(question: str) -> dict:
    response = requests.post(
        API_URL,
        json={"session_id": str(uuid.uuid4()), "question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def build_dataset() -> Dataset:
    records = []
    for question in QUESTIONS:
        result = run_rag(question)
        records.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": result.get("contexts", []),
            }
        )
    return Dataset.from_list(records)


def write_scores_csv(scores: dict) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "score"])
        writer.writeheader()
        for metric_name, value in scores.items():
            writer.writerow({"metric": metric_name, "score": value})


if __name__ == "__main__":
    dataset = build_dataset()

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_bge_embeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    write_scores_csv(scores)

    print("RAGAS scores written to evaluation/ragas_scores.csv")
    print(scores)
