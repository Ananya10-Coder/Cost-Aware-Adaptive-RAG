
import json
from pathlib import Path

from src.controllers.controller_utils import router_features


DATA = Path("data/processed")
RESULTS = Path("results")

HYBRID_PATH = RESULTS / "hybrid_results.json"
QRELS_PATH = DATA / "qrels_dev.jsonl"
QUESTIONS_PATH = DATA / "questions_dev.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def metrics(ranking, relevant):
    top = ranking[:5]
    hits = set(top) & relevant

    recall = len(hits) / len(relevant) if relevant else 0.0

    mrr = next(
        (
            1.0 / rank
            for rank, pid in enumerate(top, 1)
            if pid in relevant
        ),
        0.0
    )

    sufficient = int(
        bool(relevant) and relevant.issubset(set(top))
    )

    return recall, mrr, sufficient


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    print("Loading input data...")

    with open(HYBRID_PATH, "r", encoding="utf-8") as f:
        hybrid = json.load(f)

    qrels = {
        row["question_id"]: set(row["supporting_passage_ids"])
        for row in load_jsonl(QRELS_PATH)
    }

    question_map = {
        row["id"]: row
        for row in load_jsonl(QUESTIONS_PATH)
    }

    router_rows = []
    sufficiency_rows = []

    summary = {
        method: {
            "recall": [],
            "mrr": [],
            "sufficient": 0
        }
        for method in ["vector", "graph", "hybrid"]
    }

    missing_questions = []
    missing_qrels = []

    print("Preparing retrieval and router training data...")

    for item in hybrid["results"]:
        qid = item["question_id"]

        if qid not in question_map:
            missing_questions.append(qid)
            continue

        if qid not in qrels or not qrels[qid]:
            missing_qrels.append(qid)
            continue

        relevant = qrels[qid]
        question = item["question"]
        question_obj = question_map[qid]

        rankings = {
            "vector": item.get("vector_ranking", []),
            "graph": item.get("graph_ranking", []),
            "hybrid": [
                x["passage_id"]
                for x in item.get("retrieved", [])
            ]
        }

        scores = {}

        for method, ranking in rankings.items():
            recall, mrr, sufficient = metrics(
                ranking,
                relevant
            )

            scores[method] = {
                "recall": recall,
                "mrr": mrr,
                "sufficient": sufficient
            }

            summary[method]["recall"].append(recall)
            summary[method]["mrr"].append(mrr)
            summary[method]["sufficient"] += sufficient

            # Sufficiency data is retained for separate experiments.
            if method in ("vector", "graph"):
                sufficiency_rows.append({
                    "question_id": qid,
                    "method": method,
                    "question": question,
                    "ranking": ranking[:5],
                    "recall_at_5": recall,
                    "mrr": mrr,
                    "sufficient": sufficient
                })

        # -----------------------------------------------------
        # Query-only router features.
        # -----------------------------------------------------

        features = router_features(question_obj)

        vector_recall = scores["vector"]["recall"]
        graph_recall = scores["graph"]["recall"]

        # -----------------------------------------------------
        # Pairwise utility target: Graph Recall@5 - Vector Recall@5.
        # Positive values favor Graph; negative values favor Vector.
        # -----------------------------------------------------

        vector_utility = vector_recall
        graph_utility = graph_recall

        # Direct classifier labels exclude exact Recall ties.
        if vector_recall > graph_recall:
            route_label = "vector"
        elif graph_recall > vector_recall:
            route_label = "graph"
        else:
            route_label = None

        router_rows.append({
            "question_id": qid,
            "question": question,
            "features": features,
            "vector_utility": vector_utility,
            "graph_utility": graph_utility,
            "route_label": route_label
        })

    # ---------------------------------------------------------
    # Validate dataset coverage.
    # ---------------------------------------------------------

    if missing_questions:
        print(
            f"WARNING: {len(missing_questions)} questions "
            "were not found in questions_dev.jsonl."
        )

    if missing_qrels:
        print(
            f"WARNING: {len(missing_qrels)} questions "
            "were missing valid qrels."
        )

    if not router_rows:
        raise ValueError(
            "No router training rows were generated."
        )

    # ---------------------------------------------------------
    # Build summary.
    # ---------------------------------------------------------

    report = {}

    for method, values in summary.items():
        report[method] = {
            "mean_recall_at_5": average(values["recall"]),
            "mean_mrr": average(values["mrr"]),
            "sufficient_questions": values["sufficient"],
            "total_questions": len(values["recall"])
        }

    valid_routes = [
        row for row in router_rows
        if row["route_label"] is not None
    ]

    report["router_training"] = {
        "total_questions": len(router_rows),
        "usable_questions": len(valid_routes),
        "vector_labels": sum(
            row["route_label"] == "vector"
            for row in valid_routes
        ),
        "graph_labels": sum(
            row["route_label"] == "graph"
            for row in valid_routes
        ),
        "tied_questions": (
            len(router_rows) - len(valid_routes)
        ),
        "target": "Recall@5 only",
        "pairwise_target": (
            "graph_utility - vector_utility"
        )
    }

    # ---------------------------------------------------------
    # Save output files.
    # ---------------------------------------------------------

    RESULTS.mkdir(exist_ok=True)

    with open(
        RESULTS / "baseline_comparison.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    with open(
        RESULTS / "router_training_data.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(router_rows, f, indent=2)

    with open(
        RESULTS / "sufficiency_training_data.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(sufficiency_rows, f, indent=2)

    # ---------------------------------------------------------
    # Print report.
    # ---------------------------------------------------------

    print("\nBASELINE COMPARISON")

    for method in ["vector", "graph", "hybrid"]:
        print(method.upper(), report[method])

    print("\nROUTER TRAINING DATA")
    print(json.dumps(report["router_training"], indent=2))

    print("\nSaved:")
    print("results/baseline_comparison.json")
    print("results/router_training_data.json")
    print("results/sufficiency_training_data.json")


if __name__ == "__main__":
    main()
