
import json
import re
import math
import spacy

from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

ROOT = Path(".")
DATA = ROOT / "data/processed"
OUT = ROOT / "indexes/graph"

CORPUS_FILE = DATA / "corpus.jsonl"
QUESTIONS_FILE = DATA / "questions_dev.jsonl"
QRELS_FILE = DATA / "qrels_dev.jsonl"

GRAPH_FILE = OUT / "adaptive_graph.json"
RESULT_FILE = ROOT / "results/graph_results.json"

TOP_K = 5
SEED_K = 5

STOPWORDS = {
    "who", "what", "where", "when", "which", "why", "how",
    "is", "was", "were", "are", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "that",
    "does", "did", "do", "has", "have", "had", "with",
    "from", "by", "whose", "whom", "it", "they", "he",
    "she", "this", "these", "those", "their", "its"
}

# Generic words are not useful graph entities.
BAD_ENTITIES = STOPWORDS | {
    "addition", "thing", "something", "someone", "everything",
    "year", "time", "day", "way", "number", "part", "group",
    "people", "person", "work", "role", "area", "kind",
    "type", "result", "fact", "information", "example"
}

# Generic relation synonyms, not question-specific answer paths.
QUERY_SYNONYMS = {
    "spouse": {"partner", "wife", "husband", "married"},
    "employer": {"company", "organization", "employed"},
    "headquartered": {"headquarters", "located", "based"},
    "owns": {"owner", "owned", "ownership"},
    "founded": {"founder", "cofounder", "established"},
    "distributor": {"distributed", "distribution"},
}


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s&'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_entities(doc):
    entities = set()

    # Named entities are the primary graph nodes.
    for ent in doc.ents:
        value = normalize(ent.text)

        if len(value) >= 3 and value not in BAD_ENTITIES:
            if ent.label_ in {
                "PERSON", "ORG", "GPE", "LOC", "FAC",
                "WORK_OF_ART", "EVENT", "PRODUCT"
            }:
                entities.add(value)

    # Keep meaningful noun phrases, but reject generic fragments.
    for chunk in doc.noun_chunks:
        value = normalize(chunk.text)
        words = tokenize(value)

        if (
            len(words) >= 2
            and len(value) >= 5
            and value not in BAD_ENTITIES
            and not all(w in BAD_ENTITIES for w in words)
        ):
            entities.add(value)

    # Remove nested generic mentions when a more specific phrase exists.
    cleaned = set(entities)

    for a in entities:
        for b in entities:
            if a != b and a in b and len(b) > len(a) + 4:
                if len(a.split()) == 1:
                    cleaned.discard(a)

    return cleaned


def build_graph(corpus, nlp):
    nodes = defaultdict(set)
    edges = []
    passage_entities = {}

    print("\nBUILDING GRAPH")

    for i, passage in enumerate(corpus.values(), 1):
        pid = passage["id"]
        text = passage["text"]

        doc = nlp(text)
        entities = extract_entities(doc)

        passage_entities[pid] = entities

        for entity in entities:
            nodes[entity].add(pid)

        # Within-passage entity relationships.
        # These are explicitly labelled co-occurrence, not
        # falsely presented as semantic relations.
        for a, b in combinations(sorted(entities), 2):
            edges.append({
                "source": a,
                "target": b,
                "relation": "co_occurs",
                "passage_id": pid
            })

        if i % 100 == 0:
            print(f"Processed {i}/{len(corpus)} passages")

    graph = {
        "nodes": [
            {
                "id": entity,
                "passage_ids": sorted(pids)
            }
            for entity, pids in nodes.items()
        ],
        "edges": edges,
        "passage_entities": {
            pid: sorted(entities)
            for pid, entities in passage_entities.items()
        }
    }

    OUT.mkdir(parents=True, exist_ok=True)

    with GRAPH_FILE.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)

    print("\nGRAPH BUILT")
    print("Passages:", len(corpus))
    print("Entity nodes:", len(nodes))
    print("Edges:", len(edges))
    print("Saved:", GRAPH_FILE)

    return graph


def expanded_query_terms(query):
    terms = set(tokenize(query))

    for word, synonyms in QUERY_SYNONYMS.items():
        if word in terms:
            terms.update(synonyms)

    return terms


def calculate_bm25(query, candidate_ids, corpus):
    query_terms = expanded_query_terms(query)

    documents = {
        pid: tokenize(corpus[pid]["title"] + " " + corpus[pid]["text"])
        for pid in candidate_ids
        if pid in corpus
    }

    if not documents:
        return {}

    N = len(documents)
    avgdl = sum(len(d) for d in documents.values()) / N
    df = Counter()

    for tokens in documents.values():
        for term in set(tokens):
            df[term] += 1

    scores = {}

    k1 = 1.5
    b = 0.75

    for pid, tokens in documents.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0

        for term in query_terms:
            frequency = tf[term]

            if not frequency:
                continue

            idf = math.log(
                1 + (N - df[term] + 0.5) /
                (df[term] + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * dl / max(avgdl, 1)
            )

            score += idf * (
                frequency * (k1 + 1) / denominator
            )

        scores[pid] = score

    return scores


def retrieve(query, question, graph, corpus):
    candidates = [
        pid for pid in question["candidate_passage_ids"]
        if pid in corpus
    ]

    if not candidates:
        return []

    candidate_set = set(candidates)

    nodes = {
        node["id"]: set(node["passage_ids"])
        for node in graph["nodes"]
    }

    passage_entities = {
        pid: set(entities)
        for pid, entities in graph["passage_entities"].items()
    }

    # Build an entity-to-passage index.
    entity_index = defaultdict(set)

    for pid, entities in passage_entities.items():
        for entity in entities:
            entity_index[entity].add(pid)

    # Stage 1: lexical retrieval.
    lexical = calculate_bm25(query, candidates, corpus)

    max_score = max(lexical.values(), default=0)

    if max_score == 0:
        return []

    normalized_lexical = {
        pid: score / max_score
        for pid, score in lexical.items()
    }

    seeds = sorted(
        candidates,
        key=lambda pid: lexical.get(pid, 0),
        reverse=True
    )[:SEED_K]

    # Stage 2: graph expansion.
    graph_scores = defaultdict(float)
    evidence = defaultdict(list)

    for seed in seeds:
        seed_entities = passage_entities.get(seed, set())

        for entity in seed_entities:
            connected = entity_index.get(entity, set())

            # Weight rare entities more than common entities.
            idf = math.log(
                1 + len(corpus) /
                (1 + len(connected))
            )

            for pid in connected:
                if pid not in candidate_set:
                    continue

                if pid == seed:
                    continue

                graph_scores[pid] += 0.25 * idf
                evidence[pid].append(
                    "shared_entity:" + entity
                )

    max_graph = max(graph_scores.values(), default=0)

    results = []

    for pid in candidates:
        lexical_score = normalized_lexical.get(pid, 0)

        graph_score = (
            graph_scores.get(pid, 0) / max_graph
            if max_graph > 0 else 0
        )

        # Lexical relevance remains dominant; graph provides
        # a complementary signal.
        final_score = (
            0.70 * lexical_score
            + 0.30 * graph_score
        )

        results.append({
            "passage_id": pid,
            "score": round(final_score, 5),
            "lexical_score": round(lexical_score, 5),
            "graph_score": round(graph_score, 5),
            "title": corpus[pid]["title"],
            "text": corpus[pid]["text"],
            "graph_evidence": evidence.get(pid, [])
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:TOP_K]


def evaluate(questions, qrels, graph, corpus):
    recalls = []
    mrrs = []
    output = []

    for question in questions[:5]:
        relevant = set(qrels.get(question["id"], []))

        results = retrieve(
            question["question"],
            question,
            graph,
            corpus
        )

        retrieved_ids = [
            r["passage_id"] for r in results
        ]

        hits = relevant.intersection(retrieved_ids)

        recall = (
            len(hits) / len(relevant)
            if relevant else 0
        )

        reciprocal_rank = 0

        for rank, pid in enumerate(retrieved_ids, 1):
            if pid in relevant:
                reciprocal_rank = 1 / rank
                break

        recalls.append(recall)
        mrrs.append(reciprocal_rank)

        print("\n" + "=" * 65)
        print("QUESTION:", question["question"])
        print("Recall@5:", round(recall, 3))
        print("MRR:", round(reciprocal_rank, 3))

        for rank, result in enumerate(results, 1):
            print(
                f"\n{rank}. {result['title']} "
                f"| Score: {result['score']}"
            )
            print("ID:", result["passage_id"])
            print("Graph evidence:", result["graph_evidence"][:4])

        output.append({
            "question_id": question["id"],
            "question": question["question"],
            "recall_at_5": recall,
            "mrr": reciprocal_rank,
            "retrieved": results
        })

    summary = {
        "questions": len(output),
        "mean_recall_at_5": (
            sum(recalls) / len(recalls) if recalls else 0
        ),
        "mean_mrr": (
            sum(mrrs) / len(mrrs) if mrrs else 0
        ),
        "results": output
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("GRAPH EVALUATION COMPLETE")
    print("Questions:", summary["questions"])
    print("Mean Recall@5:", round(summary["mean_recall_at_5"], 4))
    print("Mean MRR:", round(summary["mean_mrr"], 4))
    print("Saved:", RESULT_FILE)


def main():
    print("Loading data...")

    corpus_list = load_jsonl(CORPUS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)
    qrels_list = load_jsonl(QRELS_FILE)

    corpus = {
        p["id"]: p for p in corpus_list
    }

    qrels = {
        q["question_id"]: q["supporting_passage_ids"]
        for q in qrels_list
    }

    print("Corpus passages:", len(corpus))
    print("Questions:", len(questions))

    nlp = spacy.load("en_core_web_sm")

    graph = build_graph(corpus, nlp)

    evaluate(
        questions,
        qrels,
        graph,
        corpus
    )


if __name__ == "__main__":
    main()
