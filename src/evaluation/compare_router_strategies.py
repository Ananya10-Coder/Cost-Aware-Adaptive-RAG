import json
import time
import numpy as np

from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import GroupKFold

from src.controllers.controller_utils import (
    load_json,
    load_jsonl,
    load_retrieval_runtime,
    router_vector,
    fit_classifier,
    fit_utility_regressor,
    vector_rank,
    graph_rank,
    retrieval_metrics,
)

RESULTS = Path("results")
OUTPUT_PATH = RESULTS / "router_strategies_comparison.json"

STRATEGIES = [
    "ridge_recall",
    "direct_classifier",
    "pairwise_utility",
]


def mean(values):
    return float(np.mean(values)) if values else 0.0


def get_metrics(ranking, relevant):
    recall, mrr, complete = retrieval_metrics(ranking, relevant)
    return {
        "recall_at_5": float(recall),
        "mrr": float(mrr),
        "complete_evidence": bool(complete),
    }


def main():
    print("Loading dataset...")

    questions = load_jsonl("data/processed/questions_dev.jsonl")
    corpus_rows = load_jsonl("data/processed/corpus.jsonl")
    qrels_rows = load_jsonl("data/processed/qrels_dev.jsonl")

    corpus = {p["id"]: p for p in corpus_rows}

    qrels = {
        q["question_id"]: set(q["supporting_passage_ids"])
        for q in qrels_rows
    }

    question_map = {q["id"]: q for q in questions}

    router_rows = load_json(RESULTS / "router_training_data.json")
    suff_rows = load_json(RESULTS / "sufficiency_training_data.json")

    router_by_qid = defaultdict(list)
    suff_by_qid = defaultdict(list)

    for row in router_rows:
        router_by_qid[row["question_id"]].append(row)

    for row in suff_rows:
        suff_by_qid[row["question_id"]].append(row)

    print("Questions:", len(questions))
    print("Router training rows:", len(router_rows))
    print("Sufficiency rows:", len(suff_rows))

    print("\nLoading retrieval runtime...")
    passages, embeddings, passage_indices, graph, model = (
        load_retrieval_runtime()
    )

    question_ids = np.asarray([q["id"] for q in questions])
    splitter = GroupKFold(n_splits=5)

    store = {
        name: {
            "recall": [],
            "mrr": [],
            "latency_ms": [],
            "complete": [],
            "routes": [],
            "correct": [],
            "regret": [],
        }
        for name in STRATEGIES
    }

    oracle_recalls = []
    oracle_mrrs = []
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(question_ids, groups=question_ids), 1
    ):
        train_ids = set(question_ids[train_idx])
        test_ids = set(question_ids[test_idx])

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold}: {len(train_ids)} train / {len(test_ids)} test")
        print("=" * 60)

        # ---------------------------------------------------------
        # 1. Train existing Ridge Recall@5 baseline.
        # ---------------------------------------------------------

        train_suff = [
            row
            for qid in train_ids
            for row in suff_by_qid[qid]
        ]

        vector_train = [
            row for row in train_suff
            if row["method"] == "vector"
        ]

        graph_train = [
            row for row in train_suff
            if row["method"] == "graph"
        ]

        def feature_matrix(rows):
            return np.vstack([
                np.asarray(
                    router_vector(question_map[row["question_id"]]),
                    dtype=float
                ).reshape(-1)
                for row in rows
            ])

        X_vector = feature_matrix(vector_train)
        X_graph = feature_matrix(graph_train)

        y_vector = np.asarray([
            retrieval_metrics(
                row["ranking"],
                qrels.get(row["question_id"], set())
            )[0]
            for row in vector_train
        ], dtype=float)

        y_graph = np.asarray([
            retrieval_metrics(
                row["ranking"],
                qrels.get(row["question_id"], set())
            )[0]
            for row in graph_train
        ], dtype=float)

        vector_model = fit_utility_regressor(X_vector, y_vector)
        graph_model = fit_utility_regressor(X_graph, y_graph)

        # ---------------------------------------------------------
        # 2. Train direct classifier.
        # Tied questions are excluded from classification.
        # ---------------------------------------------------------

        classifier_rows = [
            row
            for qid in train_ids
            for row in router_by_qid[qid]
            if row["route_label"] is not None
        ]

        X_classifier = feature_matrix(classifier_rows)

        y_classifier = np.asarray([
            row["route_label"] for row in classifier_rows
        ])

        classifier = fit_classifier(
            X_classifier,
            y_classifier,
            balanced=True
        )

        # ---------------------------------------------------------
        # 3. Train pairwise utility regressor.
        # Target = Graph utility - Vector utility.
        # Tied questions are retained with target zero.
        # ---------------------------------------------------------

        pairwise_rows = [
            row
            for qid in train_ids
            for row in router_by_qid[qid]
        ]

        X_pairwise = feature_matrix(pairwise_rows)

        y_pairwise = np.asarray([
            float(row["graph_utility"]) - float(row["vector_utility"])
            for row in pairwise_rows
        ], dtype=float)

        pairwise_model = fit_utility_regressor(
            X_pairwise,
            y_pairwise
        )

        print("Ridge training rows:", len(vector_train), len(graph_train))
        print("Classifier training rows:", len(classifier_rows))
        print("Pairwise training rows:", len(pairwise_rows))

        # ---------------------------------------------------------
        # 4. Evaluate on held-out questions.
        # Both rankings are computed for evaluation baselines.
        # Each simulated strategy selects only ONE ranking.
        # ---------------------------------------------------------

        for qid in question_ids[test_idx]:
            qid = str(qid)
            question = question_map[qid]
            relevant = qrels.get(qid, set())

            features = np.asarray(
                router_vector(question),
                dtype=float
            ).reshape(1, -1)

            # Run both retrieval baselines once.
            start = time.perf_counter()
            vector_ids = vector_rank(
                question, embeddings, passage_indices, model
            )
            vector_time = time.perf_counter() - start

            start = time.perf_counter()
            graph_ids = graph_rank(question, graph, corpus)
            graph_time = time.perf_counter() - start

            vector_metrics = get_metrics(vector_ids, relevant)
            graph_metrics = get_metrics(graph_ids, relevant)

            # Oracle is evaluation-only.
            oracle_recall = max(
                vector_metrics["recall_at_5"],
                graph_metrics["recall_at_5"]
            )
            oracle_mrr = max(
                vector_metrics["mrr"],
                graph_metrics["mrr"]
            )

            oracle_recalls.append(oracle_recall)
            oracle_mrrs.append(oracle_mrr)

            if vector_metrics["recall_at_5"] > graph_metrics["recall_at_5"]:
                oracle_method = "vector"
            elif graph_metrics["recall_at_5"] > vector_metrics["recall_at_5"]:
                oracle_method = "graph"
            else:
                oracle_method = "tie"

            # -----------------------------------------------------
            # Strategy A: Existing Ridge Recall@5 router.
            # -----------------------------------------------------

            start = time.perf_counter()

            predicted_vector = float(
                vector_model.predict(features)[0]
            )
            predicted_graph = float(
                graph_model.predict(features)[0]
            )

            ridge_inference_time = time.perf_counter() - start

            ridge_route = (
                "vector"
                if predicted_vector >= predicted_graph
                else "graph"
            )

            # -----------------------------------------------------
            # Strategy B: Direct supervised classifier.
            # -----------------------------------------------------

            start = time.perf_counter()
            classifier_route = str(classifier.predict(features)[0])
            classifier_inference_time = time.perf_counter() - start

            # -----------------------------------------------------
            # Strategy C: Pairwise utility regression.
            # -----------------------------------------------------

            start = time.perf_counter()

            predicted_difference = float(
                pairwise_model.predict(features)[0]
            )

            pairwise_inference_time = time.perf_counter() - start

            pairwise_route = (
                "graph" if predicted_difference > 0 else "vector"
            )

            predictions = {
                "ridge_recall": (
                    ridge_route,
                    ridge_inference_time
                ),
                "direct_classifier": (
                    classifier_route,
                    classifier_inference_time
                ),
                "pairwise_utility": (
                    pairwise_route,
                    pairwise_inference_time
                ),
            }

            question_result = {
                "question_id": qid,
                "fold": fold,
                "oracle_method": oracle_method,
                "vector_recall": vector_metrics["recall_at_5"],
                "graph_recall": graph_metrics["recall_at_5"],
                "strategies": {},
            }

            for strategy, (route, inference_time) in predictions.items():

                # Exactly one retrieval family is selected.
                if route == "vector":
                    selected_metrics = vector_metrics
                    selected_time = vector_time
                else:
                    selected_metrics = graph_metrics
                    selected_time = graph_time

                total_latency = (
                    inference_time + selected_time
                ) * 1000

                regret = (
                    oracle_recall - selected_metrics["recall_at_5"]
                )

                correct = (
                    None if oracle_method == "tie"
                    else route == oracle_method
                )

                values = store[strategy]

                values["recall"].append(
                    selected_metrics["recall_at_5"]
                )
                values["mrr"].append(selected_metrics["mrr"])
                values["latency_ms"].append(total_latency)
                values["complete"].append(
                    int(selected_metrics["complete_evidence"])
                )
                values["routes"].append(route)
                values["regret"].append(regret)

                if correct is not None:
                    values["correct"].append(int(correct))

                question_result["strategies"][strategy] = {
                    "route": route,
                    "recall_at_5": selected_metrics["recall_at_5"],
                    "mrr": selected_metrics["mrr"],
                    "complete_evidence": selected_metrics["complete_evidence"],
                    "latency_ms": total_latency,
                    "retrieval_calls": 1,
                    "oracle_regret": regret,
                    "decisive_correct": correct,
                }

            fold_details.append(question_result)

        print("Completed fold", fold)

    # -------------------------------------------------------------
    # 5. Aggregate results.
    # -------------------------------------------------------------

    summary = {}

    for strategy in STRATEGIES:
        values = store[strategy]

        summary[strategy] = {
            "mean_recall_at_5": mean(values["recall"]),
            "mean_mrr": mean(values["mrr"]),
            "mean_latency_ms": mean(values["latency_ms"]),
            "mean_retrieval_calls": 1.0,
            "complete_evidence_count": int(sum(values["complete"])),
            "questions": len(values["recall"]),
            "route_distribution": dict(Counter(values["routes"])),
            "decisive_accuracy": mean(values["correct"]),
            "decisive_questions": len(values["correct"]),
            "mean_oracle_regret": mean(values["regret"]),
        }

    oracle_summary = {
        "mean_recall_at_5": mean(oracle_recalls),
        "mean_mrr": mean(oracle_mrrs),
        "questions": len(oracle_recalls),
    }

    output = {
        "evaluation": "5-fold grouped out-of-fold evaluation",
        "dataset_questions": len(questions),
        "feature_count": len(router_vector(questions[0]).flatten()),
        "routing_constraint": (
            "Exactly one retrieval family per question; "
            "no fallback, no RRF, no result merging."
        ),
        "training_objectives": {
            "ridge_recall": "Predict Vector and Graph Recall@5 independently.",
            "direct_classifier": (
                "Predict route_label; tied questions excluded."
            ),
            "pairwise_utility": (
                "Predict graph_utility minus vector_utility; "
                "ties retained."
            ),
        },
        "results": summary,
        "oracle": oracle_summary,
        "question_details": fold_details,
    }

    RESULTS.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 65)
    print("ROUTER STRATEGIES COMPARISON")
    print("=" * 65)

    for strategy, metrics in summary.items():
        print("\n", strategy.upper())
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\nORACLE")
    for key, value in oracle_summary.items():
        print(f"{key}: {value}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
