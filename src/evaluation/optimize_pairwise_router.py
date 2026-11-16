import json
import time
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_regression

from src.controllers.controller_utils import (
    load_json,
    load_jsonl,
    load_retrieval_runtime,
    router_vector,
    fit_utility_regressor,
    vector_rank,
    graph_rank,
    rrf_rank,
    retrieval_metrics,
)

RESULTS = Path("results")
OUTPUT_PATH = RESULTS / "pairwise_optimization.json"

OUTER_FOLDS = 5
INNER_FOLDS = 4
LATENCY_REPEATS = 3
BOOTSTRAP_ITERATIONS = 10000
SEED = 42

FEATURE_COUNTS = [5, 10, 15, 22]


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def metrics_for(ranking, relevant):
    recall, mrr, complete = retrieval_metrics(ranking, relevant)
    return {
        "recall_at_5": float(recall),
        "mrr": float(mrr),
        "complete_evidence": bool(complete),
    }


def bootstrap_ci(a, b, rng):
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(differences)

    if n == 0:
        return {}

    samples = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, n, size=n)
        samples[i] = np.mean(differences[indices])

    lower, upper = np.percentile(samples, [2.5, 97.5])

    return {
        "mean_difference": float(np.mean(differences)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "interpretation": (
            "improvement" if lower > 0
            else "regression" if upper < 0
            else "inconclusive"
        ),
        "positive_questions": int(np.sum(differences > 0)),
        "negative_questions": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
    }


def make_selector(k, n_features):
    if k >= n_features:
        return None

    return SelectKBest(
        score_func=f_regression,
        k=k
    )


def fit_model(X, y, k):
    selector = make_selector(k, X.shape[1])

    if selector is None:
        X_selected = X
    else:
        X_selected = selector.fit_transform(X, y)

    model = fit_utility_regressor(X_selected, y)

    return selector, model


def predict_model(selector, model, X):
    X_selected = X if selector is None else selector.transform(X)
    return np.asarray(model.predict(X_selected), dtype=float)


def tune_threshold(predictions, qids, vector_recall, graph_recall):
    """
    Select threshold using only inner out-of-fold training predictions.

    Graph is selected when predicted(Graph utility - Vector utility)
    exceeds the threshold.

    Primary objective: maximize training OOF Recall@5.
    Tie-break: prefer more Graph selections because Graph is cheaper
    in the current measured retrieval implementation.
    """

    values = np.asarray([
        predictions[qid] for qid in qids
    ], dtype=float)

    candidates = np.quantile(
        values, np.linspace(0, 1, 21)
    ).tolist()

    candidates.extend([
        0.0,
        float(np.min(values) - 1e-8),
        float(np.max(values) + 1e-8),
    ])

    candidates = sorted(set(float(x) for x in candidates))

    best_threshold = 0.0
    best_recall = -1.0
    best_graph_fraction = -1.0

    for threshold in candidates:
        selected_recalls = []
        graph_count = 0

        for qid in qids:
            use_graph = predictions[qid] > threshold

            if use_graph:
                graph_count += 1
                selected_recalls.append(graph_recall[qid])
            else:
                selected_recalls.append(vector_recall[qid])

        score = mean(selected_recalls)
        graph_fraction = graph_count / len(qids)

        if (
            score > best_recall + 1e-12
            or (
                abs(score - best_recall) <= 1e-12
                and graph_fraction > best_graph_fraction
            )
        ):
            best_recall = score
            best_graph_fraction = graph_fraction
            best_threshold = threshold

    return {
        "threshold": float(best_threshold),
        "training_oof_recall_at_5": float(best_recall),
        "training_oof_graph_fraction": float(best_graph_fraction),
        "candidate_count": len(candidates),
    }


def main():
    rng = np.random.default_rng(SEED)

    print("Loading data...")

    questions = load_jsonl("data/processed/questions_dev.jsonl")
    corpus_rows = load_jsonl("data/processed/corpus.jsonl")
    qrels_rows = load_jsonl("data/processed/qrels_dev.jsonl")

    corpus = {p["id"]: p for p in corpus_rows}

    qrels = {
        row["question_id"]: set(row["supporting_passage_ids"])
        for row in qrels_rows
    }

    question_map = {q["id"]: q for q in questions}
    qids = [q["id"] for q in questions]

    router_rows = load_json(RESULTS / "router_training_data.json")
    suff_rows = load_json(RESULTS / "sufficiency_training_data.json")

    router_by_qid = {}
    for row in router_rows:
        qid = row["question_id"]

        if qid in router_by_qid:
            raise ValueError(
                f"Expected one router row per question, duplicate: {qid}"
            )

        router_by_qid[qid] = row

    suff_by_qid = defaultdict(dict)

    for row in suff_rows:
        qid = row["question_id"]
        method = row["method"]

        if method in suff_by_qid[qid]:
            raise ValueError(
                f"Duplicate sufficiency row: {qid}, {method}"
            )

        suff_by_qid[qid][method] = row

    missing = [
        qid for qid in qids
        if qid not in router_by_qid
        or "vector" not in suff_by_qid[qid]
        or "graph" not in suff_by_qid[qid]
        or qid not in qrels
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} questions have missing training/evaluation data."
        )

    # Query-level feature matrix. No retrieval result enters these features.
    X_all = np.vstack([
        np.asarray(
            router_vector(question_map[qid]),
            dtype=float
        ).reshape(-1)
        for qid in qids
    ])

    n_features = X_all.shape[1]

    if max(FEATURE_COUNTS) > n_features:
        raise ValueError(
            f"Requested {max(FEATURE_COUNTS)} features, "
            f"but only {n_features} exist."
        )

    row_index = {qid: i for i, qid in enumerate(qids)}

    # Pairwise training target.
    y_all = np.asarray([
        float(router_by_qid[qid]["graph_utility"])
        - float(router_by_qid[qid]["vector_utility"])
        for qid in qids
    ], dtype=float)

    # Training-only threshold tuning labels.
    vector_recall = {}
    graph_recall = {}

    for qid in qids:
        vector_recall[qid] = retrieval_metrics(
            suff_by_qid[qid]["vector"]["ranking"],
            qrels[qid]
        )[0]

        graph_recall[qid] = retrieval_metrics(
            suff_by_qid[qid]["graph"]["ranking"],
            qrels[qid]
        )[0]

    # Current pairwise model is retained as the zero-threshold baseline.
    variants = ["pairwise_zero"] + [
        f"tuned_k{k}" for k in FEATURE_COUNTS
    ]

    records = {name: [] for name in variants}
    vector_records = []
    graph_records = []
    hybrid_records = []
    question_details = []
    threshold_details = []

    print("\nQuestions:", len(qids))
    print("Features:", n_features)
    print("Outer folds:", OUTER_FOLDS)
    print("Inner folds:", INNER_FOLDS)
    print("Variants:", variants)

    print("\nLoading retrieval runtime...")
    passages, embeddings, passage_indices, graph, model = (
        load_retrieval_runtime()
    )

    print("Warming up...")
    vector_rank(questions[0], embeddings, passage_indices, model)
    graph_rank(questions[0], graph, corpus)

    outer_splitter = GroupKFold(n_splits=OUTER_FOLDS)

    for fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(
            np.asarray(qids),
            groups=np.asarray(qids)
        ),
        1
    ):
        train_qids = [qids[i] for i in outer_train_idx]
        test_qids = [qids[i] for i in outer_test_idx]

        train_indices = np.asarray([
            row_index[qid] for qid in train_qids
        ])

        test_indices = np.asarray([
            row_index[qid] for qid in test_qids
        ])

        X_train = X_all[train_indices]
        y_train = y_all[train_indices]

        print(f"\n{'=' * 65}")
        print(f"OUTER FOLD {fold}/{OUTER_FOLDS}")
        print(f"Training: {len(train_qids)} | Test: {len(test_qids)}")
        print("=" * 65)

        # ---------------------------------------------------------
        # Generate inner out-of-fold predictions for each feature
        # count. These predictions are used only to tune thresholds.
        # ---------------------------------------------------------

        inner_oof = {
            k: {} for k in FEATURE_COUNTS
        }

        inner_splitter = GroupKFold(n_splits=INNER_FOLDS)

        for inner_train_idx, inner_val_idx in inner_splitter.split(
            np.asarray(train_qids),
            groups=np.asarray(train_qids)
        ):
            inner_train_qids = [
                train_qids[i] for i in inner_train_idx
            ]

            inner_val_qids = [
                train_qids[i] for i in inner_val_idx
            ]

            inner_train_global = np.asarray([
                row_index[qid] for qid in inner_train_qids
            ])

            inner_val_global = np.asarray([
                row_index[qid] for qid in inner_val_qids
            ])

            X_inner_train = X_all[inner_train_global]
            X_inner_val = X_all[inner_val_global]

            y_inner_train = y_all[inner_train_global]

            for k in FEATURE_COUNTS:
                selector, inner_model = fit_model(
                    X_inner_train,
                    y_inner_train,
                    k
                )

                predictions = predict_model(
                    selector,
                    inner_model,
                    X_inner_val
                )

                for qid, prediction in zip(
                    inner_val_qids, predictions
                ):
                    inner_oof[k][qid] = float(prediction)

        # ---------------------------------------------------------
        # Tune thresholds from inner OOF predictions only.
        # ---------------------------------------------------------

        tuned = {}

        for k in FEATURE_COUNTS:
            if len(inner_oof[k]) != len(train_qids):
                raise RuntimeError(
                    f"Incomplete inner OOF predictions for k={k}"
                )

            tuned[k] = tune_threshold(
                inner_oof[k],
                train_qids,
                vector_recall,
                graph_recall
            )

            print(
                f"k={k}: threshold={tuned[k]['threshold']:.6f}, "
                f"OOF Recall@5={tuned[k]['training_oof_recall_at_5']:.5f}, "
                f"Graph fraction={tuned[k]['training_oof_graph_fraction']:.3f}"
            )

        # ---------------------------------------------------------
        # Fit final outer-fold models using all outer training data.
        # ---------------------------------------------------------

        final_models = {}

        for k in FEATURE_COUNTS:
            selector, fitted_model = fit_model(
                X_train,
                y_train,
                k
            )

            test_predictions = predict_model(
                selector,
                fitted_model,
                X_all[test_indices]
            )

            final_models[k] = {
                "selector": selector,
                "model": fitted_model,
                "predictions": {
                    qid: float(pred)
                    for qid, pred in zip(test_qids, test_predictions)
                },
            }

        # Existing pairwise policy: all 22 features, threshold zero.
        # The k=22 fitted model is also the zero-threshold model.
        # ---------------------------------------------------------

        fold_thresholds = {
            "pairwise_zero": 0.0,
            **{
                f"tuned_k{k}": tuned[k]["threshold"]
                for k in FEATURE_COUNTS
            }
        }

        threshold_details.append({
            "fold": fold,
            "thresholds": fold_thresholds,
            "training_oof": tuned,
        })

        # ---------------------------------------------------------
        # Evaluate each held-out question.
        # ---------------------------------------------------------

        for qid in test_qids:
            question = question_map[qid]
            relevant = qrels[qid]

            # Independent baseline timings.
            vector_times = []
            graph_times = []
            hybrid_times = []

            vector_ids = None
            graph_ids = None
            hybrid_ids = None

            for repetition in range(LATENCY_REPEATS):
                start = time.perf_counter()
                current_vector = vector_rank(
                    question, embeddings, passage_indices, model
                )
                vector_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                current_graph = graph_rank(
                    question, graph, corpus
                )
                graph_times.append(time.perf_counter() - start)

                hybrid_start = time.perf_counter()
                current_hybrid = rrf_rank(
                    current_vector, current_graph
                )
                hybrid_times.append(
                    time.perf_counter() - hybrid_start
                )

                if repetition == 0:
                    vector_ids = current_vector
                    graph_ids = current_graph
                    hybrid_ids = current_hybrid

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

            for name, ranking in baseline_rankings.items():
                metrics = metrics_for(ranking, relevant)
                baseline_metrics[name] = metrics

                record = {
                    "question_id": qid,
                    "fold": fold,
                    "recall_at_5": metrics["recall_at_5"],
                    "mrr": metrics["mrr"],
                    "complete_evidence": metrics["complete_evidence"],
                    "latency_ms": float(
                        np.median(baseline_times[name]) * 1000
                    ),
                }

                if name == "vector":
                    vector_records.append(record)
                elif name == "graph":
                    graph_records.append(record)
                else:
                    hybrid_records.append(record)

            vector_r = baseline_metrics["vector"]["recall_at_5"]
            graph_r = baseline_metrics["graph"]["recall_at_5"]
            oracle_r = max(vector_r, graph_r)

            oracle_method = (
                "vector" if vector_r > graph_r
                else "graph" if graph_r > vector_r
                else "tie"
            )

            detail = {
                "question_id": qid,
                "fold": fold,
                "vector_recall": vector_r,
                "graph_recall": graph_r,
                "oracle_method": oracle_method,
                "systems": {},
            }

            # Each adaptive variant independently executes its route.
            for variant in variants:
                if variant == "pairwise_zero":
                    k = 22
                    threshold = 0.0
                else:
                    k = int(variant.split("k")[1])
                    threshold = tuned[k]["threshold"]

                fitted = final_models[k]
                selector = fitted["selector"]
                fitted_model = fitted["model"]

                predicted_difference = fitted["predictions"][qid]

                expected_route = (
                    "graph"
                    if predicted_difference > threshold
                    else "vector"
                )

                measured_times = []
                rankings = []

                for repetition in range(LATENCY_REPEATS):
                    start = time.perf_counter()

                    # Include feature selection and prediction in routing time.
                    feature_row = X_all[row_index[qid]].reshape(1, -1)

                    route_prediction = predict_model(
                        selector,
                        fitted_model,
                        feature_row
                    )[0]

                    route = (
                        "graph"
                        if route_prediction > threshold
                        else "vector"
                    )

                    if route == "graph":
                        ranking = graph_rank(question, graph, corpus)
                    else:
                        ranking = vector_rank(
                            question, embeddings, passage_indices, model
                        )

                    measured_times.append(
                        time.perf_counter() - start
                    )
                    rankings.append(ranking)

                if route != expected_route:
                    raise RuntimeError(
                        f"Route changed between predictions: {variant}, {qid}"
                    )

                selected_metrics = metrics_for(rankings[0], relevant)

                regret = (
                    oracle_r - selected_metrics["recall_at_5"]
                )

                decisive_correct = (
                    None if oracle_method == "tie"
                    else route == oracle_method
                )

                record = {
                    "question_id": qid,
                    "fold": fold,
                    "route": route,
                    "threshold": float(threshold),
                    "feature_count": k,
                    "predicted_utility_difference": float(
                        predicted_difference
                    ),
                    "recall_at_5": selected_metrics["recall_at_5"],
                    "mrr": selected_metrics["mrr"],
                    "complete_evidence": selected_metrics["complete_evidence"],
                    "latency_ms": float(
                        np.median(measured_times) * 1000
                    ),
                    "retrieval_calls": 1,
                    "oracle_regret": regret,
                    "decisive_correct": decisive_correct,
                }

                records[variant].append(record)
                detail["systems"][variant] = record

            question_details.append(detail)

        print(f"Fold {fold} complete.")

    # -------------------------------------------------------------
    # Summaries.
    # -------------------------------------------------------------

    def summarize(rows, adaptive=False):
        result = {
            "questions": len(rows),
            "mean_recall_at_5": mean([
                r["recall_at_5"] for r in rows
            ]),
            "mean_mrr": mean([
                r["mrr"] for r in rows
            ]),
            "mean_latency_ms": mean([
                r["latency_ms"] for r in rows
            ]),
            "median_latency_ms": float(np.median([
                r["latency_ms"] for r in rows
            ])),
            "p95_latency_ms": float(np.percentile([
                r["latency_ms"] for r in rows
            ], 95)),
            "complete_evidence_count": int(sum(
                r["complete_evidence"] for r in rows
            )),
        }

        if adaptive:
            correct = [
                r["decisive_correct"]
                for r in rows
                if r["decisive_correct"] is not None
            ]

            result.update({
                "route_distribution": dict(Counter(
                    r["route"] for r in rows
                )),
                "decisive_accuracy": mean(correct),
                "decisive_questions": len(correct),
                "mean_oracle_regret": mean([
                    r["oracle_regret"] for r in rows
                ]),
                "mean_retrieval_calls": 1.0,
            })

        return result

    summaries = {
        "vector": summarize(vector_records),
        "graph": summarize(graph_records),
        "hybrid": summarize(hybrid_records),
    }

    for variant in variants:
        summaries[variant] = summarize(
            records[variant], adaptive=True
        )

    # -------------------------------------------------------------
    # Paired bootstrap against Vector and Hybrid.
    # -------------------------------------------------------------

    baseline_maps = {
        "vector": {r["question_id"]: r for r in vector_records},
        "hybrid": {r["question_id"]: r for r in hybrid_records},
    }

    comparisons = {}

    for variant in variants:
        variant_map = {
            r["question_id"]: r for r in records[variant]
        }

        comparisons[variant] = {}

        for baseline in ["vector", "hybrid"]:
            baseline_map = baseline_maps[baseline]
            common = sorted(set(variant_map) & set(baseline_map))

            comparisons[variant][baseline] = bootstrap_ci(
                [
                    variant_map[qid]["recall_at_5"]
                    for qid in common
                ],
                [
                    baseline_map[qid]["recall_at_5"]
                    for qid in common
                ],
                rng
            )

    # -------------------------------------------------------------
    # Feature selection frequencies.
    # -------------------------------------------------------------

    selection_counts = Counter()

    for fold in range(1, OUTER_FOLDS + 1):
        model_info = final_models if fold == OUTER_FOLDS else None
        # Feature selection is fitted separately within each fold.
        # Recover selected indices from the stored fold model below.

    # Reconstruct selection frequencies from outer-fold training sets.
    # This is descriptive only; it does not affect test predictions.
    outer_splitter = GroupKFold(n_splits=OUTER_FOLDS)

    for train_idx, test_idx in outer_splitter.split(
        np.asarray(qids), groups=np.asarray(qids)
    ):
        X_fold = X_all[train_idx]
        y_fold = y_all[train_idx]

        for k in FEATURE_COUNTS:
            selector = make_selector(k, n_features)

            if selector is not None:
                selector.fit(X_fold, y_fold)
                for feature_idx in selector.get_support(indices=True):
                    selection_counts[(k, int(feature_idx))] += 1
            else:
                for feature_idx in range(n_features):
                    selection_counts[(k, feature_idx)] += 1

    feature_selection_frequency = {
        str(k): {
            str(feature_idx): int(selection_counts[(k, feature_idx)])
            for feature_idx in range(n_features)
        }
        for k in FEATURE_COUNTS
    }

    output = {
        "experiment": "Nested threshold tuning and feature selection",
        "dataset": "MuSiQue dev sampled dataset",
        "questions": len(qids),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "feature_count": n_features,
        "feature_counts_tested": FEATURE_COUNTS,
        "latency_repetitions": LATENCY_REPEATS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "primary_metric": "Recall@5",
        "threshold_policy": (
            "Graph if predicted graph-minus-vector utility exceeds "
            "the threshold. Threshold selected using inner out-of-fold "
            "training predictions only. Primary objective is training "
            "OOF Recall@5; ties prefer more Graph selections."
        ),
        "feature_selection": (
            "SelectKBest with f_regression fitted only on each "
            "training partition."
        ),
        "routing_constraint": (
            "Every adaptive variant executes exactly one retrieval "
            "family per question. No RRF or fallback."
        ),
        "results": summaries,
        "paired_bootstrap_vs_baselines": comparisons,
        "thresholds_by_outer_fold": threshold_details,
        "feature_selection_frequency": feature_selection_frequency,
        "question_details": question_details,
    }

    RESULTS.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 75)
    print("PAIRWISE OPTIMIZATION RESULTS")
    print("=" * 75)

    for name, result in summaries.items():
        print(f"\n{name.upper()}")
        for key, value in result.items():
            print(f"{key}: {value}")

    print("\n" + "=" * 75)
    print("PAIRED BOOTSTRAP COMPARISONS")
    print("=" * 75)

    for variant, baselines in comparisons.items():
        for baseline, result in baselines.items():
            print(f"\n{variant} vs {baseline}")
            print(
                "Difference:",
                f"{result['mean_difference']:+.5f}"
            )
            print(
                "95% CI:",
                f"[{result['ci_95_lower']:+.5f}, "
                f"{result['ci_95_upper']:+.5f}]"
            )
            print("Conclusion:", result["interpretation"])

    print("\nTHRESHOLDS BY FOLD")
    for fold in threshold_details:
        print(
            f"Fold {fold['fold']}:",
            json.dumps(fold["thresholds"])
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
