import json
import numpy as np

from pathlib import Path
from collections import defaultdict

from src.controllers.controller_utils import (
    load_jsonl,
    load_retrieval_runtime,
    router_features,
    vector_rank,
    graph_rank,
    retrieval_metrics,
)

OUTPUT = Path("results/graph_retrieval_diagnostics.json")


def mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def get_metrics(ranking, relevant):
    recall, mrr, complete = retrieval_metrics(ranking, relevant)

    return {
        "recall_at_5": float(recall),
        "mrr": float(mrr),
        "complete_evidence": bool(complete),
    }


def length_group(question):
    count = len(question.get("question", "").split())

    if count <= 10:
        return "short_1_10"
    if count <= 20:
        return "medium_11_20"
    return "long_21_plus"


def support_group(count):
    if count == 2:
        return "two_supporting_passages"
    if count == 3:
        return "three_supporting_passages"
    return "four_or_more_supporting_passages"


def graph_statistics(graph):
    result = {"type": str(type(graph))}

    for name in ("number_of_nodes", "number_of_edges"):
        method = getattr(graph, name, None)

        if callable(method):
            try:
                result[name] = int(method())
            except Exception:
                pass

    nodes = getattr(graph, "nodes", None)
    edges = getattr(graph, "edges", None)

    if nodes is not None:
        try:
            result["nodes_length"] = len(nodes)
        except Exception:
            pass

    if edges is not None:
        try:
            result["edges_length"] = len(edges)
        except Exception:
            pass

    if isinstance(graph, dict):
        result["top_level_keys"] = [
            str(key) for key in list(graph.keys())[:20]
        ]
        result["top_level_size"] = len(graph)

    return result


def summarize_group(rows):
    if not rows:
        return {
            "questions": 0,
            "vector_recall": 0.0,
            "graph_recall": 0.0,
            "graph_advantage": 0.0,
            "graph_win_count": 0,
            "vector_win_count": 0,
            "ties": 0,
        }

    return {
        "questions": len(rows),
        "vector_recall": mean([
            r["vector_recall"] for r in rows
        ]),
        "graph_recall": mean([
            r["graph_recall"] for r in rows
        ]),
        "graph_advantage": mean([
            r["graph_minus_vector_recall"] for r in rows
        ]),
        "graph_win_count": sum(
            r["graph_minus_vector_recall"] > 0 for r in rows
        ),
        "vector_win_count": sum(
            r["graph_minus_vector_recall"] < 0 for r in rows
        ),
        "ties": sum(
            r["graph_minus_vector_recall"] == 0 for r in rows
        ),
    }


def main():
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

    print("Questions:", len(questions))
    print("Corpus:", len(corpus))

    print("\nLoading retrieval runtime...")
    passages, embeddings, passage_indices, graph, model = (
        load_retrieval_runtime()
    )

    print("Graph structure:", graph_statistics(graph))

    print("\nWarming up...")
    vector_rank(questions[0], embeddings, passage_indices, model)
    graph_rank(questions[0], graph, corpus)

    details = []
    feature_values = defaultdict(list)

    support_groups = defaultdict(list)
    length_groups = defaultdict(list)

    graph_total_hits = 0
    vector_total_hits = 0
    graph_unique_hits = 0
    vector_unique_hits = 0

    print("\nEvaluating Vector and Graph...")

    for index, question in enumerate(questions, 1):
        qid = question["id"]

        if qid not in qrels:
            raise ValueError(f"Missing qrels for {qid}")

        relevant = qrels[qid]

        vector_ids = vector_rank(
            question, embeddings, passage_indices, model
        )

        graph_ids = graph_rank(
            question, graph, corpus
        )

        vector_metrics = get_metrics(vector_ids, relevant)
        graph_metrics = get_metrics(graph_ids, relevant)

        vector_top5 = set(vector_ids[:5])
        graph_top5 = set(graph_ids[:5])

        vector_hits = vector_top5 & relevant
        graph_hits = graph_top5 & relevant

        graph_unique = graph_hits - vector_top5
        vector_unique = vector_hits - graph_top5

        graph_total_hits += len(graph_hits)
        vector_total_hits += len(vector_hits)
        graph_unique_hits += len(graph_unique)
        vector_unique_hits += len(vector_unique)

        raw_features = router_features(question)

        numeric_features = {}

        for name, value in raw_features.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_features[str(name)] = float(value)
                feature_values[str(name)].append(float(value))

        delta = (
            graph_metrics["recall_at_5"]
            - vector_metrics["recall_at_5"]
        )

        record = {
            "question_id": qid,
            "question": question.get("question", ""),
            "supporting_passage_count": len(relevant),
            "question_length": len(
                question.get("question", "").split()
            ),
            "length_group": length_group(question),
            "support_group": support_group(len(relevant)),
            "vector_recall": vector_metrics["recall_at_5"],
            "graph_recall": graph_metrics["recall_at_5"],
            "vector_mrr": vector_metrics["mrr"],
            "graph_mrr": graph_metrics["mrr"],
            "graph_minus_vector_recall": delta,
            "vector_top5": list(vector_ids[:5]),
            "graph_top5": list(graph_ids[:5]),
            "relevant_passages": sorted(relevant),
            "vector_support_hits": sorted(vector_hits),
            "graph_support_hits": sorted(graph_hits),
            "graph_unique_support_hits": sorted(graph_unique),
            "vector_unique_support_hits": sorted(vector_unique),
            "features": numeric_features,
        }

        details.append(record)

        support_groups[record["support_group"]].append(record)
        length_groups[record["length_group"]].append(record)

        if index % 100 == 0:
            print(f"Processed {index}/{len(questions)}")

    # ---------------------------------------------------------
    # Overall results.
    # ---------------------------------------------------------

    vector_recall = mean([
        r["vector_recall"] for r in details
    ])

    graph_recall = mean([
        r["graph_recall"] for r in details
    ])

    graph_wins = sum(
        r["graph_minus_vector_recall"] > 0 for r in details
    )

    vector_wins = sum(
        r["graph_minus_vector_recall"] < 0 for r in details
    )

    ties = len(details) - graph_wins - vector_wins

    overall = {
        "questions": len(details),
        "vector_recall_at_5": vector_recall,
        "graph_recall_at_5": graph_recall,
        "graph_minus_vector_recall": graph_recall - vector_recall,
        "graph_wins": graph_wins,
        "vector_wins": vector_wins,
        "ties": ties,
        "vector_total_support_hits": vector_total_hits,
        "graph_total_support_hits": graph_total_hits,
        "graph_unique_support_hits": graph_unique_hits,
        "vector_unique_support_hits": vector_unique_hits,
    }

    # ---------------------------------------------------------
    # Analyze by supporting passage count.
    # ---------------------------------------------------------

    support_analysis = {
        name: summarize_group(rows)
        for name, rows in support_groups.items()
    }

    # ---------------------------------------------------------
    # Analyze by query length.
    # ---------------------------------------------------------

    length_analysis = {
        name: summarize_group(rows)
        for name, rows in length_groups.items()
    }

    # ---------------------------------------------------------
    # Correlation of features with Graph's Recall@5 advantage.
    # This is exploratory, not causal evidence.
    # ---------------------------------------------------------

    feature_analysis = []

    for name in sorted(feature_values):
        values = [
            r["features"].get(name, 0.0)
            for r in details
        ]

        deltas = [
            r["graph_minus_vector_recall"]
            for r in details
        ]

        graph_win_labels = [
            int(r["graph_minus_vector_recall"] > 0)
            for r in details
        ]

        graph_winner_values = [
            r["features"].get(name, 0.0)
            for r in details
            if r["graph_minus_vector_recall"] > 0
        ]

        non_graph_winner_values = [
            r["features"].get(name, 0.0)
            for r in details
            if r["graph_minus_vector_recall"] <= 0
        ]

        feature_analysis.append({
            "feature": name,
            "correlation_with_recall_difference": safe_correlation(
                values, deltas
            ),
            "correlation_with_graph_win": safe_correlation(
                values, graph_win_labels
            ),
            "mean_when_graph_wins": mean(graph_winner_values),
            "mean_when_graph_does_not_win": mean(
                non_graph_winner_values
            ),
        })

    feature_analysis.sort(
        key=lambda x: abs(
            x["correlation_with_recall_difference"]
        ),
        reverse=True
    )

    # ---------------------------------------------------------
    # Most useful and most problematic examples.
    # ---------------------------------------------------------

    graph_advantage_cases = sorted(
        details,
        key=lambda r: r["graph_minus_vector_recall"],
        reverse=True
    )[:20]

    vector_advantage_cases = sorted(
        details,
        key=lambda r: r["graph_minus_vector_recall"]
    )[:20]

    output = {
        "experiment": "Graph versus Vector retrieval diagnostics",
        "questions": len(details),
        "graph_structure": graph_statistics(graph),
        "overall": overall,
        "by_supporting_passage_count": support_analysis,
        "by_question_length": length_analysis,
        "feature_correlations": feature_analysis,
        "largest_graph_advantages": graph_advantage_cases,
        "largest_vector_advantages": vector_advantage_cases,
        "question_details": details,
        "interpretation_warning": (
            "Feature correlations are exploratory and do not establish "
            "causality. The same benchmark is used to inspect retrieval "
            "outcomes, so these analyses must not be used as independent "
            "test-set evidence."
        ),
    }

    OUTPUT.parent.mkdir(exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("GRAPH RETRIEVAL DIAGNOSTICS")
    print("=" * 70)

    print("\nOVERALL")
    print(json.dumps(overall, indent=2))

    print("\nBY SUPPORTING PASSAGE COUNT")
    print(json.dumps(support_analysis, indent=2))

    print("\nBY QUESTION LENGTH")
    print(json.dumps(length_analysis, indent=2))

    print("\nTOP FEATURES CORRELATED WITH GRAPH ADVANTAGE")
    for row in feature_analysis[:12]:
        print(
            row["feature"],
            "| delta correlation:",
            round(row["correlation_with_recall_difference"], 4),
            "| Graph wins:",
            round(row["mean_when_graph_wins"], 4),
            "| Others:",
            round(row["mean_when_graph_does_not_win"], 4),
        )

    print("\nTOP GRAPH ADVANTAGE EXAMPLES")
    for row in graph_advantage_cases[:10]:
        print(
            row["question_id"],
            "| delta:",
            row["graph_minus_vector_recall"],
            "| question:",
            row["question"]
        )

    print("\nTOP VECTOR ADVANTAGE EXAMPLES")
    for row in vector_advantage_cases[:10]:
        print(
            row["question_id"],
            "| delta:",
            row["graph_minus_vector_recall"],
            "| question:",
            row["question"]
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
