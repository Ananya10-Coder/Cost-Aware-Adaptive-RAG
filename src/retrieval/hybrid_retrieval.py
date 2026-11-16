import json
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.graph.graph_pipeline import retrieve as graph_retrieve

ROOT = Path(".")
DATA_DIR = ROOT / "data/processed"
VECTOR_DIR = ROOT / "indexes/vector"
GRAPH_FILE = ROOT / "indexes/graph/adaptive_graph.json"
RESULT_FILE = ROOT / "results/hybrid_results.json"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 5
RRF_K = 60
MAX_QUESTIONS = 500  # Change to 50 after the initial test


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    scores = {}

    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)


def vector_rank(question, passages, passage_embeddings, model):
    candidates = set(question["candidate_passage_ids"])

    candidate_indices = [
        i for i, passage in enumerate(passages)
        if passage["id"] in candidates
    ]

    if not candidate_indices:
        return []

    query = (
        "Represent this sentence for searching relevant passages: "
        + question["question"]
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")

    candidate_matrix = passage_embeddings[candidate_indices]
    scores = candidate_matrix @ query_embedding

    ranked_positions = np.argsort(scores)[::-1][:TOP_K]

    return [
        passages[candidate_indices[pos]]["id"]
        for pos in ranked_positions
    ]


def evaluate(questions, qrels, corpus, graph, passages,
             passage_embeddings, model):

    output = []
    recalls = []
    mrrs = []

    for i, question in enumerate(questions[:MAX_QUESTIONS], 1):
        qid = question["id"]
        relevant = set(qrels.get(qid, []))

        # Retrieve independently using both methods.
        vector_ids = vector_rank(
            question, passages, passage_embeddings, model
        )

        graph_results = graph_retrieve(
            question["question"], question, graph, corpus
        )
        graph_ids = [
            item["passage_id"] for item in graph_results
        ]

        # Fuse rankings. RRF does not compare incompatible raw scores.
        hybrid_ids = reciprocal_rank_fusion(
            [vector_ids, graph_ids]
        )[:TOP_K]

        hits = relevant.intersection(hybrid_ids)
        recall = len(hits) / len(relevant) if relevant else 0.0

        reciprocal_rank = 0.0
        for rank, pid in enumerate(hybrid_ids, start=1):
            if pid in relevant:
                reciprocal_rank = 1.0 / rank
                break

        recalls.append(recall)
        mrrs.append(reciprocal_rank)

        retrieved = []
        for rank, pid in enumerate(hybrid_ids, start=1):
            passage = corpus[pid]
            retrieved.append({
                "rank": rank,
                "passage_id": pid,
                "title": passage["title"],
                "score": round(
                    sum(
                        1.0 / (RRF_K + r)
                        for ranking in [vector_ids, graph_ids]
                        for r, ranked_pid in enumerate(ranking, 1)
                        if ranked_pid == pid
                    ), 6
                ),
                "text": passage["text"]
            })

        output.append({
            "question_id": qid,
            "question": question["question"],
            "recall_at_5": recall,
            "mrr": reciprocal_rank,
            "vector_ranking": vector_ids,
            "graph_ranking": graph_ids,
            "retrieved": retrieved
        })

        print(f"\n{i}/{min(len(questions), MAX_QUESTIONS)}")
        print("Question:", question["question"])
        print("Vector:", vector_ids)
        print("Graph:", graph_ids)
        print("Hybrid:", hybrid_ids)
        print("Recall@5:", round(recall, 3), "| MRR:",
              round(reciprocal_rank, 3))

    summary = {
        "method": "Hybrid RAG with Reciprocal Rank Fusion",
        "questions": len(output),
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "mean_recall_at_5": (
            float(np.mean(recalls)) if recalls else 0.0
        ),
        "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "results": output
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("HYBRID EVALUATION COMPLETE")
    print("Questions:", summary["questions"])
    print("Mean Recall@5:", round(summary["mean_recall_at_5"], 4))
    print("Mean MRR:", round(summary["mean_mrr"], 4))
    print("Saved:", RESULT_FILE)


def main():
    print("Loading dataset and indexes...")

    corpus_list = load_jsonl(DATA_DIR / "corpus.jsonl")
    questions = load_jsonl(DATA_DIR / "questions_dev.jsonl")
    qrels_list = load_jsonl(DATA_DIR / "qrels_dev.jsonl")

    corpus = {p["id"]: p for p in corpus_list}
    qrels = {
        q["question_id"]: q["supporting_passage_ids"]
        for q in qrels_list
    }

    with (VECTOR_DIR / "passages.json").open(encoding="utf-8") as f:
        passages = json.load(f)

    passage_embeddings = np.load(VECTOR_DIR / "embeddings.npy")
    passage_embeddings = passage_embeddings.astype("float32")

    with GRAPH_FILE.open(encoding="utf-8") as f:
        graph = json.load(f)

    if len(passages) != len(passage_embeddings):
        raise ValueError(
            "Vector passage metadata and embeddings have different lengths."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Corpus:", len(corpus))
    print("Questions:", len(questions))
    print("Vector passages:", len(passages))
    print("Graph nodes:", len(graph["nodes"]))

    evaluate(
        questions, qrels, corpus, graph,
        passages, passage_embeddings, model
    )


if __name__ == "__main__":
    main()
