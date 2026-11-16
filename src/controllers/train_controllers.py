
import json
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.controllers.controller_utils import (
    load_json,
    load_jsonl,
    sufficiency_features,
    fit_classifier,
    fit_utility_regressor,
    ROUTER_KEYS,
)

RESULTS = Path("results")
MODEL_DIR = Path("indexes/controllers")


def cross_validate_pairwise(X, y, groups):
    predictions = np.zeros(len(y), dtype=float)

    splitter = GroupKFold(n_splits=5)

    for train_idx, test_idx in splitter.split(X, y, groups):
        model = fit_utility_regressor(
            X[train_idx], y[train_idx]
        )
        predictions[test_idx] = model.predict(X[test_idx])

    decisive = np.abs(y) > 1e-9

    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "decisive_route_accuracy": float(
            np.mean(
                (predictions[decisive] > 0) == (y[decisive] > 0)
            )
        ) if decisive.any() else 0.0,
        "training_examples": len(y),
        "tied_examples": int(np.sum(np.abs(y) <= 1e-9)),
    }


def cross_validate_classifier(X, y, groups):
    predictions = np.full(len(y), None, dtype=object)
    splitter = GroupKFold(n_splits=5)

    for train_idx, test_idx in splitter.split(X, y, groups):
        model = fit_classifier(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    valid = np.asarray([p is not None for p in predictions])
    actual = y[valid]
    predicted = np.asarray(predictions[valid], dtype=y.dtype)

    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro",
                     zero_division=0)
        )
    }


def main():
    router_rows = load_json(
        RESULTS / "router_training_data.json"
    )
    suff_rows = load_json(
        RESULTS / "sufficiency_training_data.json"
    )
    corpus_rows = load_jsonl(
        "data/processed/corpus.jsonl"
    )
    question_rows = load_jsonl(
        "data/processed/questions_dev.jsonl"
    )

    corpus = {p["id"]: p for p in corpus_rows}
    questions = {q["id"]: q for q in question_rows}

    # --------------------------------------------------
    # PAIRWISE UTILITY MODEL
    # Train on ALL questions, including ties.
    # Target = Graph Recall@5 - Vector Recall@5.
    # --------------------------------------------------

    X_pairwise = np.asarray([
        [row["features"][key] for key in ROUTER_KEYS]
        for row in router_rows
    ], dtype=np.float64)

    y_pairwise = np.asarray([
        row["graph_utility"] - row["vector_utility"]
        for row in router_rows
    ], dtype=np.float64)

    groups_pairwise = np.asarray([
        row["question_id"] for row in router_rows
    ])

    print("\nPAIRWISE TRAINING DATA")
    print("Examples:", len(y_pairwise))
    print("Graph better:", int(np.sum(y_pairwise > 0)))
    print("Vector better:", int(np.sum(y_pairwise < 0)))
    print("Tied:", int(np.sum(np.isclose(y_pairwise, 0))))

    pairwise_metrics = cross_validate_pairwise(
        X_pairwise,
        y_pairwise,
        groups_pairwise
    )

    print("\nPAIRWISE GROUPED CROSS-VALIDATION")
    print(pairwise_metrics)

    pairwise_model = fit_utility_regressor(
        X_pairwise,
        y_pairwise
    )

    # --------------------------------------------------
    # SUFFICIENCY MODEL
    # Retained for analysis; not needed by the
    # one-retriever adaptive inference path.
    # --------------------------------------------------

    X_suff = []
    y_suff = []
    groups_suff = []

    for row in suff_rows:
        question = questions[row["question_id"]]

        features = sufficiency_features(
            question,
            row["ranking"],
            row["method"],
            corpus
        )[0]

        X_suff.append(features)
        y_suff.append(row["sufficient"])
        groups_suff.append(row["question_id"])

    X_suff = np.asarray(X_suff, dtype=np.float64)
    y_suff = np.asarray(y_suff, dtype=int)
    groups_suff = np.asarray(groups_suff)

    suff_metrics = cross_validate_classifier(
        X_suff,
        y_suff,
        groups_suff
    )

    suff_model = fit_classifier(X_suff, y_suff)

    # --------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pairwise_model,
        MODEL_DIR / "pairwise_utility_model.joblib"
    )

    joblib.dump(
        suff_model,
        MODEL_DIR / "sufficiency_model.joblib"
    )

    metrics = {
        "pairwise": pairwise_metrics,
        "sufficiency": suff_metrics,
        "pairwise_training_examples": len(y_pairwise),
        "sufficiency_training_examples": len(y_suff),
        "pairwise_target": "graph_recall_at_5 - vector_recall_at_5"
    }

    with open(
        RESULTS / "controller_metrics.json", "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nCONTROLLERS TRAINED")
    print(
        "Pairwise model:",
        MODEL_DIR / "pairwise_utility_model.joblib"
    )
    print(
        "Sufficiency model:",
        MODEL_DIR / "sufficiency_model.joblib"
    )
    print(
        "Metrics:",
        RESULTS / "controller_metrics.json"
    )


if __name__ == "__main__":
    main()
