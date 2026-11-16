import json
import re
import joblib
import numpy as np

from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(".")
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"

HYBRID_FILE = RESULTS / "hybrid_results.json"
CORPUS_FILE = DATA / "corpus.jsonl"
QRELS_FILE = DATA / "qrels_dev.jsonl"

MODEL_FILE = ROOT / "indexes/sufficiency_model.joblib"
METRICS_FILE = RESULTS / "sufficiency_metrics.json"

METHODS = ["vector", "graph", "hybrid"]

STOPWORDS = {
    "who", "what", "where", "when", "which", "why", "how",
    "is", "was", "were", "are", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "that",
    "does", "did", "do", "has", "have", "had", "with",
    "from", "by", "whose", "whom", "it", "they", "he",
    "she", "this", "these", "those", "their", "its"
}


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def get_rankings(result):
    return {
        "vector": result.get("vector_ranking", []),
        "graph": result.get("graph_ranking", []),
        "hybrid": [
            item["passage_id"]
            for item in result.get("retrieved", [])
        ]
    }


def extract_features(question, ranking, method, corpus):
    query_terms = set(tokenize(question["question"]))
    candidate_count = len(question.get("candidate_passage_ids", []))

    overlaps = []
    retrieved_terms = set()
    document_lengths = []

    for pid in ranking[:5]:
        passage = corpus.get(pid)

        if not passage:
            continue

        text = passage.get("title", "") + " " + passage.get("text", "")
        terms = set(tokenize(text))

        retrieved_terms.update(terms)
        document_lengths.append(len(terms))

        overlap = (
            len(query_terms.intersection(terms)) / len(query_terms)
            if query_terms else 0.0
        )
        overlaps.append(overlap)

    if not overlaps:
        overlaps = [0.0]

    # Rank-weighted lexical coverage: higher-ranked passages matter more.
    weighted_overlap = sum(
        overlap / rank
        for rank, overlap in enumerate(overlaps, 1)
    )

    query_coverage = (
        len(query_terms.intersection(retrieved_terms)) / len(query_terms)
        if query_terms else 0.0
    )

    features = [
        len(query_terms),
        candidate_count,
        len(ranking[:5]),
        overlaps[0],
        max(overlaps),
        float(np.mean(overlaps)),
        float(np.std(overlaps)),
        weighted_overlap,
        query_coverage,
        float(np.mean(document_lengths)) if document_lengths else 0.0,
        int(method == "vector"),
        int(method == "graph"),
        int(method == "hybrid")
    ]

    return features


def main():
    print("Loading evaluation data...")

    hybrid_data = load_json(HYBRID_FILE)
    corpus_list = load_jsonl(CORPUS_FILE)
    qrels_list = load_jsonl(QRELS_FILE)

    corpus = {p["id"]: p for p in corpus_list}
    qrels = {
        q["question_id"]: set(q["supporting_passage_ids"])
        for q in qrels_list
    }

    questions = {
        item["question_id"]: item
        for item in hybrid_data["results"]
    }

    X = []
    y = []
    groups = []
    methods = []

    print("Building training examples...")

    for qid, question in questions.items():
        relevant = qrels.get(qid, set())

        if not relevant:
            continue

        rankings = get_rankings(question)

        for method in METHODS:
            ranking = rankings[method]

            features = extract_features(
                question, ranking, method, corpus
            )

            # Strict evidence-completeness label.
            label = int(relevant.issubset(set(ranking[:5])))

            X.append(features)
            y.append(label)
            groups.append(qid)
            methods.append(method)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    print("\nTRAINING DATA")
    print("Examples:", len(y))
    print("Unique questions:", len(set(groups)))
    print("Sufficient:", int(y.sum()))
    print("Insufficient:", int(len(y) - y.sum()))

    if len(set(y)) < 2:
        raise ValueError(
            "Only one sufficiency class was generated. "
            "The predictor cannot learn from this dataset."
        )

    # Grouped cross-validation prevents examples from the same
    # question appearing in both training and validation folds.
    splitter = GroupKFold(n_splits=5)

    predictions = np.full(len(y), -1, dtype=int)
    fold_scores = []

    print("\n5-FOLD GROUPED CROSS-VALIDATION")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), 1
    ):
        if len(set(y[train_idx])) < 2:
            print(f"Fold {fold}: skipped (one training class)")
            continue

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

        fold_accuracy = accuracy_score(
            y[test_idx], predictions[test_idx]
        )

        fold_scores.append(fold_accuracy)

        print(
            f"Fold {fold}: accuracy = {fold_accuracy:.3f}, "
            f"validation examples = {len(test_idx)}"
        )

    valid = predictions != -1

    if not valid.any():
        raise ValueError("No valid cross-validation predictions.")

    metrics = {
        "method": "Logistic Regression",
        "evaluation": "5-fold grouped cross-validation",
        "examples": int(len(y)),
        "unique_questions": int(len(set(groups))),
        "sufficient_examples": int(y.sum()),
        "insufficient_examples": int(len(y) - y.sum()),
        "evaluated_examples": int(valid.sum()),
        "accuracy": float(accuracy_score(y[valid], predictions[valid])),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[valid], predictions[valid])
        ),
        "precision": float(
            precision_score(y[valid], predictions[valid], zero_division=0)
        ),
        "recall": float(
            recall_score(y[valid], predictions[valid], zero_division=0)
        ),
        "f1": float(
            f1_score(y[valid], predictions[valid], zero_division=0)
        ),
        "fold_accuracies": fold_scores
    }

    print("\nCROSS-VALIDATION RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Fit the deployable model on all available examples.
    final_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    final_model.fit(X, y)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_FILE)

    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSUFFICIENCY PREDICTOR TRAINED")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)


if __name__ == "__main__":
    main()
