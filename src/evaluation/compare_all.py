import json
import time
import numpy as np

from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold


from src.controllers.controller_utils import (
    load_json, load_jsonl, load_retrieval_runtime,
    router_vector, sufficiency_features,
    positive_probability, fit_classifier,
    fit_utility_regressor,
    vector_rank, graph_rank, rrf_rank,
    retrieval_metrics
)

RESULTS = Path("results")


def mean(values):
    return float(np.mean(values)) if values else 0.0


def metrics_for(ranking, relevant):
    recall, mrr, complete = retrieval_metrics(ranking, relevant)
    return {
        "recall_at_5": recall,
        "mrr": mrr,
        "complete_evidence": bool(complete)
    }


def add_result(store, method, ranking, relevant, elapsed, calls):
    metrics = metrics_for(ranking, relevant)

    store[method]["recall"].append(metrics["recall_at_5"])
    store[method]["mrr"].append(metrics["mrr"])
    store[method]["latency_ms"].append(elapsed * 1000)
    store[method]["calls"].append(calls)
    store[method]["sufficient"].append(
        int(metrics["complete_evidence"])
    )


def main():
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

    router_rows = [
        r for r in router_rows if r["route_label"] is not None
    ]

    router_by_qid = defaultdict(list)
    suff_by_qid = defaultdict(list)

    for row in router_rows:
        router_by_qid[row["question_id"]].append(row)

    for row in suff_rows:
        suff_by_qid[row["question_id"]].append(row)

    runtime = load_retrieval_runtime()
    passages, embeddings, passage_indices, graph, model = runtime

    question_ids = np.asarray([q["id"] for q in questions])
    splitter = GroupKFold(n_splits=5)

    store = defaultdict(lambda: {
        "recall": [],
        "mrr": [],
        "latency_ms": [],
        "calls": [],
        "sufficient": []
    })

    adaptive_details = []
    oracle_details = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(question_ids, groups=question_ids), 1
    ):
        train_ids = set(question_ids[train_idx])
        test_ids = set(question_ids[test_idx])

        train_suff = [
            r for qid in train_ids for r in suff_by_qid[qid]
        ]

        vector_train_rows = [
            r for r in train_suff if r["method"] == "vector"
        ]
        graph_train_rows = [
            r for r in train_suff if r["method"] == "graph"
        ]

        X_vector = np.vstack([
            np.asarray(
                router_vector(question_map[r["question_id"]]),
                dtype=float
            ).reshape(-1)
            for r in vector_train_rows
        ])

        X_graph = np.vstack([
            np.asarray(
                router_vector(question_map[r["question_id"]]),
                dtype=float
            ).reshape(-1)
            for r in graph_train_rows
        ])

        y_vector = np.asarray([
            retrieval_metrics(
                r["ranking"],
                qrels.get(r["question_id"], set())
            )[0]
            for r in vector_train_rows
        ], dtype=float)

        y_graph = np.asarray([
            retrieval_metrics(
                r["ranking"],
                qrels.get(r["question_id"], set())
            )[0]
            for r in graph_train_rows
        ], dtype=float)

        vector_recall_model = fit_utility_regressor(
            X_vector, y_vector
        )

        graph_recall_model = fit_utility_regressor(
            X_graph, y_graph
        )

        print(f"\nFOLD {fold}: testing {len(test_ids)} questions")

        for qid in question_ids[test_idx]:
            question = question_map[qid]
            relevant = qrels.get(qid, set())

            # Baseline 1: Vector.
            start = time.perf_counter()
            v_ids = vector_rank(
                question, embeddings, passage_indices, model
            )
            vector_time = time.perf_counter() - start

            add_result(
                store, "vector", v_ids, relevant, vector_time, 1
            )

            # Baseline 2: Graph.
            start = time.perf_counter()
            g_ids = graph_rank(question, graph, corpus)
            graph_time = time.perf_counter() - start

            add_result(
                store, "graph", g_ids, relevant, graph_time, 1
            )

            # Baseline 3: Hybrid (always both retrievers + RRF).
            start = time.perf_counter()
            h_ids = rrf_rank(v_ids, g_ids)
            rrf_time = time.perf_counter() - start

            hybrid_time = vector_time + graph_time + rrf_time

            add_result(
                store, "hybrid", h_ids, relevant, hybrid_time, 2
            )

            # Proposed Adaptive RAG.
            # Timer includes routing inference and exactly one retrieval.
            adaptive_start = time.perf_counter()

            query_features = np.asarray(
                router_vector(question), dtype=float
            ).reshape(1, -1)

            predicted_vector_recall = float(
                vector_recall_model.predict(query_features)[0]
            )

            predicted_graph_recall = float(
                graph_recall_model.predict(query_features)[0]
            )

            initial_route = (
                "vector"
                if predicted_vector_recall >= predicted_graph_recall
                else "graph"
            )

            retrieval_start = time.perf_counter()

            if initial_route == "vector":
                first_ids = vector_rank(
                    question, embeddings, passage_indices, model
                )
            else:
                first_ids = graph_rank(question, graph, corpus)

            first_retrieval_time = time.perf_counter() - retrieval_start
            adaptive_time = time.perf_counter() - adaptive_start

            # Metrics and oracle diagnostics are deliberately outside
            # the Adaptive latency measurement.
            first_metrics = metrics_for(first_ids, relevant)

            final_ids = first_ids
            final_method = initial_route
            final_probability = None

            first_probability = None
            first_utility_prediction = None
            expected_gain_prediction = None

            calls = 1
            fallback = False
            alternative = None
            alternative_ids = None
            alternative_probability = None
            alternative_retrieval_time = None
            alternative_metrics = None
            alternative_selected = False
            alternative_utility_prediction = None

            final_metrics = metrics_for(final_ids, relevant)

            add_result(
                store, "adaptive", final_ids, relevant,
                adaptive_time, calls
            )

            # Evaluation-only oracle diagnostics. The router never
            # receives these gold-label comparisons.
            vector_metrics = metrics_for(v_ids, relevant)
            graph_metrics = metrics_for(g_ids, relevant)

            vector_recall = vector_metrics["recall_at_5"]
            graph_recall = graph_metrics["recall_at_5"]

            oracle_recall = max(vector_recall, graph_recall)
            oracle_mrr = max(
                vector_metrics["mrr"],
                graph_metrics["mrr"]
            )

            if vector_recall > graph_recall:
                oracle_method = "vector"
            elif graph_recall > vector_recall:
                oracle_method = "graph"
            else:
                oracle_method = "tie"

            router_metrics = (
                vector_metrics
                if initial_route == "vector"
                else graph_metrics
            )

            router_correct = (
                None if oracle_method == "tie"
                else initial_route == oracle_method
            )

            oracle_details.append({
                "question_id": str(qid),
                "fold": fold,
                "router_method": initial_route,
                "oracle_method": oracle_method,
                "vector_recall_at_5": vector_recall,
                "graph_recall_at_5": graph_recall,
                "oracle_recall_at_5": oracle_recall,
                "router_recall_at_5":
                    router_metrics["recall_at_5"],
                "router_recall_regret":
                    oracle_recall - router_metrics["recall_at_5"],
                "oracle_mrr": oracle_mrr,
                "router_mrr": router_metrics["mrr"],
                "router_correct": router_correct
            })

            recall_delta = None
            mrr_delta = None
            evidence_delta = None



            # Compare the two retrieval attempts using gold labels.
            # This is diagnostic evaluation only; gold labels do not
            # participate in the Adaptive selection decision.
            if alternative_metrics is not None:
                recall_delta = (
                    alternative_metrics["recall_at_5"]
                    - first_metrics["recall_at_5"]
                )
                mrr_delta = (
                    alternative_metrics["mrr"] - first_metrics["mrr"]
                )
                evidence_delta = (
                    int(alternative_metrics["complete_evidence"])
                    - int(first_metrics["complete_evidence"])
                )
            else:
                recall_delta = None
                mrr_delta = None
                evidence_delta = None

            adaptive_details.append({
                "question_id": str(qid),
                "fold": fold,

                "initial_route": initial_route,
                "predicted_vector_recall": predicted_vector_recall,
                "predicted_graph_recall": predicted_graph_recall,
                "initial_retrieval_calls": 1,
                "initial_retrieval_time_ms":
                    first_retrieval_time * 1000,
                "initial_sufficiency_probability":
                    first_probability,
                "predicted_improvement": expected_gain_prediction,
                "initial_retrieved_ids": first_ids,
                "initial_metrics": first_metrics,

                "fallback_used": fallback,
                "alternative_method": alternative,
                "alternative_retrieval_time_ms":
                    alternative_retrieval_time,
                "alternative_sufficiency_probability":
                    alternative_probability,
                "alternative_retrieved_ids": alternative_ids,
                "alternative_metrics": alternative_metrics,

                "alternative_selected": alternative_selected,
                "final_method": final_method,
                "final_sufficiency_probability": final_probability,
                "final_retrieved_ids": final_ids,
                "final_metrics": final_metrics,

                "retrieval_calls": calls,
                "adaptive_total_latency_ms": adaptive_time * 1000,

                "alternative_minus_initial_recall": recall_delta,
                "alternative_minus_initial_mrr": mrr_delta,
                "alternative_minus_initial_complete_evidence":
                    evidence_delta,

                "fallback_improved_recall":
                    recall_delta > 0 if recall_delta is not None else None,
                "fallback_improved_mrr":
                    mrr_delta > 0 if mrr_delta is not None else None,
                "fallback_improved_complete_evidence":
                    evidence_delta > 0
                    if evidence_delta is not None else None,
                
                "initial_predicted_utility": first_utility_prediction,
                "alternative_predicted_utility": alternative_utility_prediction,
            })

    summary = {}

    for method, values in store.items():
        summary[method] = {
            "mean_recall_at_5": mean(values["recall"]),
            "mean_mrr": mean(values["mrr"]),
            "mean_latency_ms": mean(values["latency_ms"]),
            "mean_retrieval_calls": mean(values["calls"]),
            "complete_evidence_count": int(sum(values["sufficient"])),
            "questions": len(values["recall"])
        }

    decisive_router_rows = [
        x for x in oracle_details
        if x["router_correct"] is not None
    ]

    oracle_summary = {
        "oracle_mean_recall_at_5": mean([
            x["oracle_recall_at_5"] for x in oracle_details
        ]),
        "oracle_mean_mrr": mean([
            x["oracle_mrr"] for x in oracle_details
        ]),
        "router_mean_recall_at_5": mean([
            x["router_recall_at_5"] for x in oracle_details
        ]),
        "router_mean_recall_regret": mean([
            x["router_recall_regret"] for x in oracle_details
        ]),
        "router_decisive_accuracy": mean([
            int(x["router_correct"])
            for x in decisive_router_rows
        ]),
        "router_decisive_questions": len(decisive_router_rows),
        "oracle_tied_questions": sum(
            x["oracle_method"] == "tie" for x in oracle_details
        ),
        "questions": len(oracle_details)
    }

    fallback_rows = [
        x for x in adaptive_details if x["fallback_used"]
    ]

    selected_rows = [
        x for x in fallback_rows if x["alternative_selected"]
    ]

    def fallback_rate(key):
        values = [
            x[key] for x in fallback_rows if x[key] is not None
        ]
        return mean([int(v) for v in values])

    summary["adaptive"].update({
        "fallback_count": len(fallback_rows),
        "fallback_rate": (
            len(fallback_rows) / len(adaptive_details)
            if adaptive_details else 0
        ),
        "alternative_selected_count": len(selected_rows),
        "alternative_selection_rate": (
            len(selected_rows) / len(fallback_rows)
            if fallback_rows else 0
        ),
        "fallback_improved_recall_count": sum(
            x["fallback_improved_recall"] is True
            for x in fallback_rows
        ),
        "fallback_improved_mrr_count": sum(
            x["fallback_improved_mrr"] is True
            for x in fallback_rows
        ),
        "fallback_improved_complete_evidence_count": sum(
            x["fallback_improved_complete_evidence"] is True
            for x in fallback_rows
        ),
        "fallback_mean_recall_delta": mean([
            x["alternative_minus_initial_recall"]
            for x in fallback_rows
            if x["alternative_minus_initial_recall"] is not None
        ]),
        "fallback_mean_mrr_delta": mean([
            x["alternative_minus_initial_mrr"]
            for x in fallback_rows
            if x["alternative_minus_initial_mrr"] is not None
        ]),
        "fallback_mean_evidence_delta": mean([
            x["alternative_minus_initial_complete_evidence"]
            for x in fallback_rows
            if x["alternative_minus_initial_complete_evidence"]
            is not None
        ])
    })

    output = {
        "evaluation": "5-fold grouped out-of-fold evaluation",
        "routing_policy": "query-only predicted Recall@5; single retrieval path",
        "questions": len(questions),
        "results": summary,
        "oracle_diagnostics": oracle_summary,
        "oracle_question_details": oracle_details,
        "adaptive_question_details": adaptive_details
    }

    RESULTS.mkdir(exist_ok=True)

    output_path = RESULTS / "all_systems_comparison.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 65)
    print("ALL SYSTEMS COMPARISON")

    for method, metrics in summary.items():
        print("\n", method.upper())
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\nORACLE DIAGNOSTICS")
    for key, value in oracle_summary.items():
        print(f"{key}: {value}")

    print("\nFALLBACK DIAGNOSTICS")
    for key, value in summary["adaptive"].items():
        if key.startswith("fallback_") or key.startswith("alternative_"):
            print(f"{key}: {value}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
