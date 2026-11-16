import json
import time
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
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
    rrf_rank,
    retrieval_metrics,
)

RESULTS = Path("results")
OUTPUT_PATH = RESULTS / "final_router_evaluation.json"

N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
LATENCY_REPEATS = 3

BASELINES = ["vector", "graph", "hybrid"]
ADAPTIVE = [
    "pairwise_utility"
]
ALL_SYSTEMS = BASELINES + ADAPTIVE


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def metrics_for(ranking, relevant):
    recall, mrr, complete = retrieval_metrics(ranking, relevant)

    return {
        "recall_at_5": float(recall),
        "mrr": float(mrr),
        "complete_evidence": bool(complete),
    }


def make_feature_matrix(rows, question_map):
    return np.vstack([
        np.asarray(
            router_vector(question_map[row["question_id"]]),
            dtype=float
        ).reshape(-1)
        for row in rows
    ])


def paired_bootstrap(a, b, rng):
    """
    Paired bootstrap confidence interval for mean(a - b).
    Each index represents the same held-out question in both systems.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Paired bootstrap requires equal nonempty arrays.")

    differences = a - b
    n = len(differences)

    bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(differences[indices])

    lower, upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    ties = int(np.sum(differences == 0))

    return {
        "mean_difference": float(np.mean(differences)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "positive_questions": wins,
        "negative_questions": losses,
        "tied_questions": ties,
        "significant_at_0_05_by_bootstrap_ci": bool(
            lower > 0 or upper < 0
        ),
        "interpretation": (
            "improvement" if lower > 0
            else "regression" if upper < 0
            else "inconclusive"
        ),
    }


def summarize_system(rows):
    recalls = [r["recall_at_5"] for r in rows]
    mrrs = [r["mrr"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    complete = [r["complete_evidence"] for r in rows]

    summary = {
        "questions": len(rows),
        "mean_recall_at_5": mean(recalls),
        "mean_mrr": mean(mrrs),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "complete_evidence_count": int(sum(complete)),
    }

    if rows and "route" in rows[0]:
        routes = [r["route"] for r in rows]
        correct = [
            r["decisive_correct"]
            for r in rows
            if r["decisive_correct"] is not None
        ]

        summary.update({
            "route_distribution": dict(Counter(routes)),
            "decisive_accuracy": mean(correct),
            "decisive_questions": len(correct),
            "mean_oracle_regret": mean([
                r["oracle_regret"] for r in rows
            ]),
            "mean_retrieval_calls_per_question": 1.0,
        })
    elif rows and rows[0]["system"] == "hybrid":
        summary["retrieval_calls_per_question"] = 2.0
    else:
        summary["retrieval_calls_per_question"] = 1.0

    return summary


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    print("Loading dataset...")

    questions = load_jsonl("data/processed/questions_dev.jsonl")
    corpus_rows = load_jsonl("data/processed/corpus.jsonl")
    qrels_rows = load_jsonl("data/processed/qrels_dev.jsonl")

    corpus = {p["id"]: p for p in corpus_rows}

    qrels = {
        row["question_id"]: set(row["supporting_passage_ids"])
        for row in qrels_rows
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

    missing_qrels = [
        q["id"] for q in questions if q["id"] not in qrels
    ]

    if missing_qrels:
        raise ValueError(
            f"{len(missing_qrels)} questions are missing qrels."
        )

    if len(questions) < N_SPLITS:
        raise ValueError("Not enough questions for GroupKFold.")

    print("Questions:", len(questions))
    print("Corpus passages:", len(corpus))
    print("Router rows:", len(router_rows))
    print("Sufficiency rows:", len(suff_rows))
    print("Latency repetitions:", LATENCY_REPEATS)

    print("\nLoading retrieval runtime...")
    passages, embeddings, passage_indices, graph, model = (
        load_retrieval_runtime()
    )

    # Warm up model and retrieval functions.
    print("Warming up retrieval models...")
    warmup_question = questions[0]
    vector_rank(
        warmup_question, embeddings, passage_indices, model
    )
    graph_rank(warmup_question, graph, corpus)

    question_ids = np.asarray([q["id"] for q in questions])
    splitter = GroupKFold(n_splits=N_SPLITS)

    records = {name: [] for name in ALL_SYSTEMS}
    oracle_records = []
    question_details = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(question_ids, groups=question_ids), 1
    ):
        train_ids = set(question_ids[train_idx])
        test_ids = set(question_ids[test_idx])

        print(f"\n{'=' * 65}")
        print(f"FOLD {fold}/{N_SPLITS}")
        print(f"Train: {len(train_ids)} | Test: {len(test_ids)}")
        print("=" * 65)

        # ---------------------------------------------------------
        # Train Ridge Recall@5 models using training-fold questions.
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

        if not vector_train or not graph_train:
            raise ValueError("Missing Vector or Graph training rows.")

        X_vector = make_feature_matrix(vector_train, question_map)
        X_graph = make_feature_matrix(graph_train, question_map)

        y_vector = np.asarray([
            retrieval_metrics(
                row["ranking"],
                qrels[row["question_id"]]
            )[0]
            for row in vector_train
        ], dtype=float)

        y_graph = np.asarray([
            retrieval_metrics(
                row["ranking"],
                qrels[row["question_id"]]
            )[0]
            for row in graph_train
        ], dtype=float)

        vector_model = fit_utility_regressor(X_vector, y_vector)
        graph_model = fit_utility_regressor(X_graph, y_graph)

        # ---------------------------------------------------------
        # Train direct classifier using non-tied training rows.
        # ---------------------------------------------------------

        classifier_rows = [
            row
            for qid in train_ids
            for row in router_by_qid[qid]
            if row["route_label"] is not None
        ]

        if not classifier_rows:
            raise ValueError("No labeled router training rows.")

        X_classifier = make_feature_matrix(
            classifier_rows, question_map
        )

        y_classifier = np.asarray([
            row["route_label"] for row in classifier_rows
        ])

        classifier = fit_classifier(
            X_classifier,
            y_classifier,
            balanced=True
        )

        # ---------------------------------------------------------
        # Train pairwise utility model using all training rows.
        # Target = Graph utility - Vector utility.
        # ---------------------------------------------------------

        pairwise_rows = [
            row
            for qid in train_ids
            for row in router_by_qid[qid]
        ]

        if not pairwise_rows:
            raise ValueError("No pairwise training rows.")

        X_pairwise = make_feature_matrix(
            pairwise_rows, question_map
        )

        y_pairwise = np.asarray([
            float(row["graph_utility"])
            - float(row["vector_utility"])
            for row in pairwise_rows
        ], dtype=float)

        pairwise_model = fit_utility_regressor(
            X_pairwise, y_pairwise
        )

        print(
            "Training rows:",
            f"Vector={len(vector_train)}",
            f"Graph={len(graph_train)}",
            f"Classifier={len(classifier_rows)}",
            f"Pairwise={len(pairwise_rows)}",
        )

        # ---------------------------------------------------------
        # Evaluate held-out questions.
        # ---------------------------------------------------------

        for qid in question_ids[test_idx]:
            qid = str(qid)
            question = question_map[qid]
            relevant = qrels[qid]

            features = np.asarray(
                router_vector(question),
                dtype=float
            ).reshape(1, -1)

            # -----------------------------------------------------
            # Baselines: independently benchmark Vector and Graph.
            # Hybrid executes both retrieval methods and RRF.
            # -----------------------------------------------------

            vector_times = []
            graph_times = []
            hybrid_times = []

            vector_ids = None
            graph_ids = None
            hybrid_ids = None

            for repetition in range(LATENCY_REPEATS):
                # Hybrid timing includes both retrieval methods and RRF.
                hybrid_start = time.perf_counter()

                start = time.perf_counter()
                current_vector_ids = vector_rank(
                    question, embeddings, passage_indices, model
                )
                vector_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                current_graph_ids = graph_rank(
                    question, graph, corpus
                )
                graph_times.append(time.perf_counter() - start)

                current_hybrid_ids = rrf_rank(
                    current_vector_ids, current_graph_ids
                )
                hybrid_times.append(
                    time.perf_counter() - hybrid_start
                )

                if repetition == 0:
                    vector_ids = current_vector_ids
                    graph_ids = current_graph_ids
                    hybrid_ids = current_hybrid_ids

            baseline_rankings = {
                "vector": vector_ids,
                "graph": graph_ids,
                "hybrid": hybrid_ids,
            }

            baseline_times = {
                "vector": vector_times,
                "graph": graph_times,
                "hybrid": hybrid_times,
            }

            baseline_metrics = {}

            for name in BASELINES:
                metrics = metrics_for(
                    baseline_rankings[name], relevant
                )

                baseline_metrics[name] = metrics

                records[name].append({
                    "question_id": qid,
                    "fold": fold,
                    "system": name,
                    "recall_at_5": metrics["recall_at_5"],
                    "mrr": metrics["mrr"],
                    "complete_evidence": metrics["complete_evidence"],
                    "latency_ms": float(
                        np.median(baseline_times[name]) * 1000
                    ),
                })

            vector_recall = baseline_metrics["vector"]["recall_at_5"]
            graph_recall = baseline_metrics["graph"]["recall_at_5"]

            oracle_recall = max(vector_recall, graph_recall)

            if vector_recall > graph_recall:
                oracle_method = "vector"
            elif graph_recall > vector_recall:
                oracle_method = "graph"
            else:
                oracle_method = "tie"

            oracle_records.append({
                "question_id": qid,
                "recall_at_5": oracle_recall,
                "mrr": max(
                    baseline_metrics["vector"]["mrr"],
                    baseline_metrics["graph"]["mrr"]
                ),
            })

            # -----------------------------------------------------
            # Determine the three routes using query-only features.
            # -----------------------------------------------------

            routes = {}

            start = time.perf_counter()
            predicted_vector = float(
                vector_model.predict(features)[0]
            )
            predicted_graph = float(
                graph_model.predict(features)[0]
            )
            ridge_inference = time.perf_counter() - start

            routes["ridge_recall"] = (
                "vector"
                if predicted_vector >= predicted_graph
                else "graph"
            )

            start = time.perf_counter()
            classifier_route = str(
                classifier.predict(features)[0]
            )
            classifier_inference = time.perf_counter() - start

            if classifier_route not in ("vector", "graph"):
                raise ValueError(
                    f"Unexpected classifier route: {classifier_route}"
                )

            routes["direct_classifier"] = classifier_route

            start = time.perf_counter()
            predicted_difference = float(
                pairwise_model.predict(features)[0]
            )
            pairwise_inference = time.perf_counter() - start

            routes["pairwise_utility"] = (
                "graph" if predicted_difference > 0 else "vector"
            )

            inference_times = {
                "ridge_recall": ridge_inference,
                "direct_classifier": classifier_inference,
                "pairwise_utility": pairwise_inference,
            }

            detail = {
                "question_id": qid,
                "fold": fold,
                "oracle_method": oracle_method,
                "oracle_recall_at_5": oracle_recall,
                "vector_recall_at_5": vector_recall,
                "graph_recall_at_5": graph_recall,
                "systems": {},
            }

            for strategy in ADAPTIVE:
                route = routes[strategy]
                selected_rankings = []
                total_times = []

                # Each repetition executes only the selected retriever.
                # Timing includes router inference and retrieval.
                for repetition in range(LATENCY_REPEATS):
                    start = time.perf_counter()

                    if strategy == "ridge_recall":
                        route_start = time.perf_counter()
                        current_vector = float(
                            vector_model.predict(features)[0]
                        )
                        current_graph = float(
                            graph_model.predict(features)[0]
                        )
                        route = (
                            "vector"
                            if current_vector >= current_graph
                            else "graph"
                        )
                        route_elapsed = (
                            time.perf_counter() - route_start
                        )

                    elif strategy == "direct_classifier":
                        route_start = time.perf_counter()
                        route = str(classifier.predict(features)[0])
                        route_elapsed = (
                            time.perf_counter() - route_start
                        )

                    else:
                        route_start = time.perf_counter()
                        difference = float(
                            pairwise_model.predict(features)[0]
                        )
                        route = "graph" if difference > 0 else "vector"
                        route_elapsed = (
                            time.perf_counter() - route_start
                        )

                    if route == "vector":
                        ranking = vector_rank(
                            question, embeddings, passage_indices, model
                        )
                    elif route == "graph":
                        ranking = graph_rank(
                            question, graph, corpus
                        )
                    else:
                        raise ValueError(
                            f"Invalid route from {strategy}: {route}"
                        )

                    elapsed = time.perf_counter() - start

                    selected_rankings.append(ranking)
                    total_times.append(elapsed)

                # All models are deterministic; ensure route consistency.
                if route != routes[strategy]:
                    raise RuntimeError(
                        f"Non-deterministic route for {strategy}, {qid}"
                    )

                selected_metrics = metrics_for(
                    selected_rankings[0], relevant
                )

                regret = (
                    oracle_recall - selected_metrics["recall_at_5"]
                )

                decisive_correct = (
                    None if oracle_method == "tie"
                    else route == oracle_method
                )

                result = {
                    "question_id": qid,
                    "fold": fold,
                    "system": strategy,
                    "route": route,
                    "recall_at_5": selected_metrics["recall_at_5"],
                    "mrr": selected_metrics["mrr"],
                    "complete_evidence": selected_metrics["complete_evidence"],
                    "latency_ms": float(
                        np.median(total_times) * 1000
                    ),
                    "retrieval_calls_per_query": 1,
                    "oracle_regret": regret,
                    "decisive_correct": decisive_correct,
                    "benchmark_repetitions": LATENCY_REPEATS,
                }

                records[strategy].append(result)

                detail["systems"][strategy] = {
                    "route": route,
                    "recall_at_5": selected_metrics["recall_at_5"],
                    "mrr": selected_metrics["mrr"],
                    "latency_ms": result["latency_ms"],
                    "oracle_regret": regret,
                    "decisive_correct": decisive_correct,
                }

            for name in BASELINES:
                detail["systems"][name] = {
                    "recall_at_5": baseline_metrics[name]["recall_at_5"],
                    "mrr": baseline_metrics[name]["mrr"],
                    "latency_ms": float(
                        np.median(baseline_times[name]) * 1000
                    ),
                }

            question_details.append(detail)

        print(f"Fold {fold} complete.")

    # -------------------------------------------------------------
    # Aggregate results.
    # -------------------------------------------------------------

    summaries = {
        name: summarize_system(records[name])
        for name in ALL_SYSTEMS
    }

    oracle_summary = {
        "mean_recall_at_5": mean([
            row["recall_at_5"] for row in oracle_records
        ]),
        "mean_mrr": mean([
            row["mrr"] for row in oracle_records
        ]),
        "questions": len(oracle_records),
    }

    # -------------------------------------------------------------
    # Paired bootstrap comparisons.
    # -------------------------------------------------------------

    by_system = {
        name: {
            row["question_id"]: row
            for row in records[name]
        }
        for name in ALL_SYSTEMS
    }

    paired_comparisons = {}

    for adaptive in ADAPTIVE:
        paired_comparisons[adaptive] = {}

        for baseline in ["vector", "hybrid"]:
            adaptive_rows = by_system[adaptive]
            baseline_rows = by_system[baseline]

            common_ids = sorted(
                set(adaptive_rows) & set(baseline_rows)
            )

            adaptive_recall = [
                adaptive_rows[qid]["recall_at_5"]
                for qid in common_ids
            ]

            baseline_recall = [
                baseline_rows[qid]["recall_at_5"]
                for qid in common_ids
            ]

            adaptive_mrr = [
                adaptive_rows[qid]["mrr"]
                for qid in common_ids
            ]

            baseline_mrr = [
                baseline_rows[qid]["mrr"]
                for qid in common_ids
            ]

            paired_comparisons[adaptive][baseline] = {
                "questions": len(common_ids),
                "recall_at_5": paired_bootstrap(
                    adaptive_recall, baseline_recall, rng
                ),
                "mrr": paired_bootstrap(
                    adaptive_mrr, baseline_mrr, rng
                ),
            }

    # -------------------------------------------------------------
    # Graph utility and routing diagnostics.
    # -------------------------------------------------------------

    graph_wins = sum(
        1 for row in question_details
        if row["graph_recall_at_5"] > row["vector_recall_at_5"]
    )

    vector_wins = sum(
        1 for row in question_details
        if row["vector_recall_at_5"] > row["graph_recall_at_5"]
    )

    retrieval_ties = len(question_details) - graph_wins - vector_wins

    routing_diagnostics = {}

    for strategy in ADAPTIVE:
        rows = records[strategy]

        graph_selected = [
            row for row in rows if row["route"] == "graph"
        ]

        vector_selected = [
            row for row in rows if row["route"] == "vector"
        ]

        routing_diagnostics[strategy] = {
            "graph_selected_count": len(graph_selected),
            "vector_selected_count": len(vector_selected),
            "graph_selected_when_graph_won": sum(
                1 for row in graph_selected
                if row["decisive_correct"] is True
            ),
            "graph_selected_when_vector_won": sum(
                1 for row in graph_selected
                if row["decisive_correct"] is False
            ),
            "vector_selected_when_vector_won": sum(
                1 for row in vector_selected
                if row["decisive_correct"] is True
            ),
            "vector_selected_when_graph_won": sum(
                1 for row in vector_selected
                if row["decisive_correct"] is False
            ),
        }

    output = {
        "experiment": "Independent adaptive retrieval evaluation",
        "dataset": "MuSiQue dev sampled dataset",
        "questions": len(questions),
        "folds": N_SPLITS,
        "latency_repetitions_per_question": LATENCY_REPEATS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_metric": "Recall@5",
        "routing_constraint": (
            "Each adaptive policy executes exactly one retrieval family. "
            "No RRF, result merging, or fallback in adaptive systems."
        ),
        "latency_method": (
            "Median of three independent measurements per question. "
            "Adaptive latency includes routing inference and the selected "
            "retrieval execution. Hybrid includes both retrieval executions "
            "and RRF."
        ),
        "results": summaries,
        "oracle": oracle_summary,
        "paired_bootstrap_comparisons": paired_comparisons,
        "retrieval_winner_counts": {
            "vector_wins": vector_wins,
            "graph_wins": graph_wins,
            "ties": retrieval_ties,
        },
        "routing_diagnostics": routing_diagnostics,
        "question_details": question_details,
    }

    RESULTS.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # -------------------------------------------------------------
    # Print report.
    # -------------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL ROUTER EVALUATION")
    print("=" * 75)

    for name in ALL_SYSTEMS:
        result = summaries[name]

        print(f"\n{name.upper()}")
        print(f"Recall@5:       {result['mean_recall_at_5']:.5f}")
        print(f"MRR:            {result['mean_mrr']:.5f}")
        print(f"Mean latency:   {result['mean_latency_ms']:.3f} ms")
        print(f"Median latency: {result['median_latency_ms']:.3f} ms")
        print(f"P95 latency:    {result['p95_latency_ms']:.3f} ms")
        print(
            "Complete:       "
            f"{result['complete_evidence_count']}/{result['questions']}"
        )

        if "route_distribution" in result:
            print("Routes:", result["route_distribution"])
            print(
                "Decisive accuracy:",
                f"{result['decisive_accuracy']:.4f}"
            )
            print(
                "Oracle regret:",
                f"{result['mean_oracle_regret']:.5f}"
            )

    print("\n" + "=" * 75)
    print("PAIRED BOOTSTRAP: ADAPTIVE MINUS BASELINE")
    print("=" * 75)

    for adaptive, comparisons in paired_comparisons.items():
        for baseline, metrics in comparisons.items():
            recall = metrics["recall_at_5"]

            print(
                f"\n{adaptive} vs {baseline} | Recall@5"
            )
            print(
                f"Difference: {recall['mean_difference']:+.5f}"
            )
            print(
                "95% CI: "
                f"[{recall['ci_95_lower']:+.5f}, "
                f"{recall['ci_95_upper']:+.5f}]"
            )
            print("Conclusion:", recall["interpretation"])

    print("\nRETRIEVAL WINNER COUNTS")
    print({
        "vector_wins": vector_wins,
        "graph_wins": graph_wins,
        "ties": retrieval_ties,
    })

    print("\nROUTING DIAGNOSTICS")
    print(json.dumps(routing_diagnostics, indent=2))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
