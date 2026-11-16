
import json
import re
import math
import sys

from pathlib import Path
from collections import defaultdict, Counter

DATA = Path("data/processed")
GRAPH_PATH = Path("indexes/graph/pilot_graph.json")

QUESTION_PATH = DATA / "questions_dev.jsonl"

STOPWORDS = {
    "who", "what", "where", "when", "which", "why", "how",
    "is", "was", "were", "are", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "that",
    "does", "did", "do", "has", "have", "had", "with",
    "from", "by", "whose", "whom"
}


def tokenize(text):
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def load_data():
    with GRAPH_PATH.open(encoding="utf-8") as f:
        graph = json.load(f)

    with (DATA / "corpus.jsonl").open(encoding="utf-8") as f:
        corpus = {
            p["id"]: p for p in map(json.loads, f)
        }

    with QUESTION_PATH.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    return graph, corpus, questions


def get_candidates(query, questions, graph, corpus):
    # Use the benchmark's candidate passages when the exact
    # evaluation question is supplied.
    for q in questions:
        if q["question"].strip().lower() == query.strip().lower():
            return [
                pid for pid in q["candidate_passage_ids"]
                if pid in corpus
            ]

    # For a custom query, use all passages represented in the graph.
    graph_pids = {
        pid
        for node in graph["nodes"]
        for pid in node.get("passage_ids", [])
    }

    return [pid for pid in graph_pids if pid in corpus]


def lexical_scores(query, candidate_ids, corpus):
    query_tokens = set(tokenize(query))

    if not query_tokens:
        return {}

    documents = {
        pid: set(tokenize(corpus[pid]["text"]))
        for pid in candidate_ids
    }

    # Inverse document frequency over the current candidate pool.
    N = max(len(documents), 1)
    df = Counter()

    for tokens in documents.values():
        for token in tokens:
            df[token] += 1

    scores = {}

    for pid, tokens in documents.items():
        score = 0.0

        for term in query_tokens:
            if term in tokens:
                idf = math.log(1 + (N - df[term] + 0.5) /
                               (df[term] + 0.5))
                score += idf

        scores[pid] = score

    return scores


def retrieve(query, top_k=5, seed_k=3, max_hops=1):
    graph, corpus, questions = load_data()

    candidates = get_candidates(query, questions, graph, corpus)

    if not candidates:
        return []

    candidate_set = set(candidates)

    nodes = {
        node["id"]: node
        for node in graph["nodes"]
    }

    # Build passage-to-node and node-to-edge mappings.
    passage_nodes = defaultdict(set)
    adjacency = defaultdict(list)

    for node_id, node in nodes.items():
        for pid in node.get("passage_ids", []):
            passage_nodes[pid].add(node_id)

    for edge in graph["edges"]:
        adjacency[edge["source"]].append(edge)

        # Treat traversal as bidirectional for retrieval.
        adjacency[edge["target"]].append({
            **edge,
            "source": edge["target"],
            "target": edge["source"]
        })

    # Stage 1: lexical seeds.
    lexical = lexical_scores(query, candidates, corpus)

    seeds = sorted(
        lexical,
        key=lexical.get,
        reverse=True
    )[:seed_k]

    scores = defaultdict(float)
    evidence = defaultdict(list)

    max_lexical = max(lexical.values(), default=0.0)

    for pid, score in lexical.items():
        if max_lexical > 0:
            scores[pid] += score / max_lexical

        if score > 0:
            evidence[pid].append("lexical_match")

    # Stage 2: graph expansion from seed passage entities.
    frontier = set()

    for pid in seeds:
        frontier.update(passage_nodes.get(pid, set()))

    visited = set(frontier)

    for hop in range(max_hops):
        next_frontier = set()

        for node_id in frontier:
            for edge in adjacency[node_id]:
                neighbor = edge["target"]
                relation = edge.get("relation", "unknown")
                edge_pid = edge.get("passage_id")

                # Typed relations are stronger than co-occurrence.
                weight = 0.60 if relation != "co_occurs" else 0.15

                if edge_pid in candidate_set:
                    scores[edge_pid] += weight
                    evidence[edge_pid].append(
                        f"{node_id} -{relation}-> {neighbor}"
                    )

                # A connected entity may occur in other passages.
                for pid in nodes.get(neighbor, {}).get(
                    "passage_ids", []
                ):
                    if pid in candidate_set:
                        scores[pid] += weight * 0.5
                        evidence[pid].append(
                            f"connected_entity:{neighbor}"
                        )

                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)

        frontier = next_frontier

    results = []

    for pid, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        if pid not in candidate_set:
            continue

        results.append({
            "passage_id": pid,
            "score": round(score, 4),
            "title": corpus[pid]["title"],
            "text": corpus[pid]["text"],
            "graph_evidence": list(dict.fromkeys(evidence[pid]))
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])

    if not query:
        query = "Who is the spouse of the Green performer?"

    results = retrieve(query)

    print("\nQUERY:", query)
    print("Retrieved:", len(results))

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']}")
        print("Passage ID:", result["passage_id"])
        print("Title:", result["title"])
        print("Evidence:", result["graph_evidence"][:5])
        print("Passage:", result["text"][:400])
