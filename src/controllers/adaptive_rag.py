
import argparse
import joblib
import numpy as np

from pathlib import Path
from src.controllers.controller_utils import (
    load_jsonl,
    load_retrieval_runtime,
    router_vector,
    vector_rank,
    graph_rank,
)

MODEL_DIR = Path("indexes/controllers")


def run_adaptive(question, corpus, runtime, pairwise_model):
    _, embeddings, passage_indices, graph, model = runtime

    # Predict Graph Recall@5 - Vector Recall@5.
    features = router_vector(question)
    predicted_difference = float(
        np.asarray(pairwise_model.predict(features)).reshape(-1)[0]
    )

    # Select the retriever predicted to achieve higher recall.
    route = "graph" if predicted_difference > 0 else "vector"

    # Execute exactly one retrieval family.
    if route == "vector":
        ranking = vector_rank(
            question, embeddings, passage_indices, model
        )
    else:
        ranking = graph_rank(question, graph, corpus)

    return {
        "final_method": route,
        "predicted_recall_difference": predicted_difference,
        "retrieval_calls": 1,
        "retrieved_ids": ranking[:5],
        "retrieved_passages": [
            corpus[pid] for pid in ranking[:5] if pid in corpus
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id")
    args = parser.parse_args()

    questions = load_jsonl("data/processed/questions_dev.jsonl")
    corpus_rows = load_jsonl("data/processed/corpus.jsonl")
    corpus = {p["id"]: p for p in corpus_rows}

    question = next(
        (q for q in questions if q["id"] == args.question_id),
        None
    ) if args.question_id else questions[0]

    if question is None:
        raise ValueError("Question ID not found")

    pairwise_model = joblib.load(
        MODEL_DIR / "pairwise_utility_model.joblib"
    )

    runtime = load_retrieval_runtime()

    result = run_adaptive(
        question, corpus, runtime, pairwise_model
    )

    print("\nADAPTIVE RAG")
    print("Question:", question["question"])
    print("Selected retriever:", result["final_method"])
    print(
        "Predicted Graph - Vector Recall:",
        round(result["predicted_recall_difference"], 4)
    )
    print("Retrieval calls:", result["retrieval_calls"])

    for i, passage in enumerate(result["retrieved_passages"], 1):
        print(f"\n{i}. {passage['title']}")
        print("ID:", passage["id"])
        print(passage["text"][:400])


if __name__ == "__main__":
    main()
