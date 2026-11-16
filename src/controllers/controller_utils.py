import json
import re
import numpy as np

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier

ROOT = Path(".")
DATA = ROOT / "data/processed"
VECTOR_DIR = ROOT / "indexes/vector"
GRAPH_FILE = ROOT / "indexes/graph/adaptive_graph.json"

ROUTER_KEYS = [
    "question_length", "unique_terms",
    "content_term_ratio", "average_token_length",
    "candidate_count",
    "has_who", "has_what", "has_where", "has_when",
    "has_which", "has_how",
    "entity_token_count", "relation_term_count",
    "comparison_term_count", "conjunction_count",
    "clause_marker_count", "pronoun_count",
    "has_multiple_entities", "has_multiple_relations",
    "has_comparison", "has_multiple_clauses",
    "has_multi_hop_cue"
]

STOPWORDS = {
    "who", "what", "where", "when", "which", "why", "how",
    "is", "was", "were", "are", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "that",
    "does", "did", "do", "has", "have", "had", "with",
    "from", "by", "whose", "whom", "it", "they", "he",
    "she", "this", "these", "those", "their", "its"
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def router_features(question):
    text = question["question"]
    lower = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+", lower)
    words = tokenize(text)
    unique_words = set(words)

    relations = {
        "spouse", "wife", "husband", "founded", "founder",
        "located", "headquartered", "associated", "connected",
        "member", "born", "married", "directed", "written",
        "produced", "played", "worked", "son", "daughter",
        "parent", "sibling", "author", "composer", "created",
        "owned", "employed", "studied", "graduated", "based"
    }
    comparisons = {
        "before", "after", "between", "both", "first",
        "later", "then", "compared", "earlier", "following",
        "greater", "larger", "smaller", "higher", "lower"
    }
    clauses = {
        "that", "which", "whose", "whom", "although",
        "because", "while", "whereas", "where", "when"
    }
    pronouns = {
        "he", "she", "they", "him", "her", "them",
        "his", "their", "hers", "theirs", "it", "its"
    }

    relation_count = sum(w in relations for w in words)
    comparison_count = sum(w in comparisons for w in words)
    conjunction_count = sum(w in {"and", "or", "but"} for w in raw_tokens)
    clause_count = sum(w in clauses for w in raw_tokens)
    pronoun_count = sum(w in pronouns for w in raw_tokens)

    capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", text)
    entities = {
        w.lower() for w in capitalized
        if w.lower() not in STOPWORDS
    }
    entity_count = len(entities)

    return {
        "question_length": len(raw_tokens),
        "unique_terms": len(unique_words),
        "content_term_ratio": len(unique_words) / max(len(raw_tokens), 1),
        "average_token_length": (
            sum(map(len, words)) / len(words) if words else 0.0
        ),
        "candidate_count": len(question.get("candidate_passage_ids", [])),

        "has_who": int(lower.startswith("who")),
        "has_what": int(lower.startswith("what")),
        "has_where": int(lower.startswith("where")),
        "has_when": int(lower.startswith("when")),
        "has_which": int(lower.startswith("which")),
        "has_how": int(lower.startswith("how")),

        "entity_token_count": entity_count,
        "relation_term_count": relation_count,
        "comparison_term_count": comparison_count,
        "conjunction_count": conjunction_count,
        "clause_marker_count": clause_count,
        "pronoun_count": pronoun_count,

        "has_multiple_entities": int(entity_count >= 2),
        "has_multiple_relations": int(relation_count >= 2),
        "has_comparison": int(comparison_count > 0),
        "has_multiple_clauses": int(clause_count + conjunction_count >= 2),
        "has_multi_hop_cue": int(
            entity_count >= 2 and (
                relation_count >= 1 or clause_count >= 1
                or conjunction_count >= 1
            )
        )
    }

def router_vector(question):
    features = router_features(question)
    return np.asarray(
        [[features[k] for k in ROUTER_KEYS]],
        dtype=np.float64
    )



_IDF_CACHE = {}


def sufficiency_features(question, ranking, method, corpus):
    """
    Predict evidence completeness using only the question and the
    passages returned by the selected retrieval method.

    No gold labels or alternative-retriever results are used.
    """

    text = question["question"]
    query_tokens = tokenize(text)
    query_terms = set(query_tokens)

    if not query_terms:
        query_terms = {""}

    # Cache corpus-level inverse document frequencies.
    cache_key = id(corpus)

    if cache_key not in _IDF_CACHE:
        document_frequency = {}

        for passage in corpus.values():
            passage_text = (
                passage.get("title", "") + " " +
                passage.get("text", "")
            )
            terms = set(tokenize(passage_text))

            for term in terms:
                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        total_documents = max(len(corpus), 1)

        _IDF_CACHE[cache_key] = {
            term: np.log(
                (total_documents + 1) / (frequency + 1)
            ) + 1.0
            for term, frequency in document_frequency.items()
        }

    idf = _IDF_CACHE[cache_key]

    retrieved_terms = set()
    passage_term_sets = []
    overlaps = []
    title_overlaps = []
    document_lengths = []

    for pid in ranking[:5]:
        passage = corpus.get(pid)

        if not passage:
            continue

        title = passage.get("title", "")
        passage_text = passage.get("text", "")

        title_terms = set(tokenize(title))
        terms = set(tokenize(title + " " + passage_text))

        passage_term_sets.append(terms)
        retrieved_terms.update(terms)
        document_lengths.append(len(terms))

        overlaps.append(
            len(query_terms.intersection(terms)) / len(query_terms)
        )

        title_overlaps.append(
            len(query_terms.intersection(title_terms)) / len(query_terms)
        )

    if not overlaps:
        overlaps = [0.0]

    if not title_overlaps:
        title_overlaps = [0.0]

    # Ranked overlap and aggregate query coverage.
    weighted_overlap = sum(
        overlap / rank
        for rank, overlap in enumerate(overlaps, 1)
    )

    query_coverage = (
        len(query_terms.intersection(retrieved_terms))
        / len(query_terms)
    )

    # IDF-weighted query coverage.
    query_weights = {
        term: idf.get(term, 1.0)
        for term in query_terms
    }

    total_query_weight = sum(query_weights.values())

    idf_coverage = (
        sum(
            weight for term, weight in query_weights.items()
            if term in retrieved_terms
        ) / total_query_weight
        if total_query_weight else 0.0
    )

    # Passage redundancy: high overlap may indicate repeated evidence.
    pairwise_jaccard = []

    for i in range(len(passage_term_sets)):
        for j in range(i + 1, len(passage_term_sets)):
            a = passage_term_sets[i]
            b = passage_term_sets[j]
            union = a.union(b)

            if union:
                pairwise_jaccard.append(
                    len(a.intersection(b)) / len(union)
                )

    redundancy = (
        float(np.mean(pairwise_jaccard))
        if pairwise_jaccard else 0.0
    )

    # Entity coverage from capitalized query expressions.
    entity_words = {
        token.lower()
        for token in __import__("re").findall(
            r"\b[A-Z][a-zA-Z0-9]+\b", text
        )
        if token.lower() not in STOPWORDS
    }

    entity_coverage = (
        len(entity_words.intersection(retrieved_terms))
        / len(entity_words)
        if entity_words else 0.0
    )

    lower = text.lower()

    relation_terms = [
        "spouse", "wife", "husband", "founded", "founder",
        "located", "headquartered", "associated", "connected",
        "member", "born", "married", "directed", "written",
        "produced", "played", "worked", "son", "daughter",
        "parent", "sibling", "author", "composer"
    ]

    comparison_terms = [
        "before", "after", "between", "both", "first",
        "later", "then", "compared", "earlier", "following"
    ]

    relation_count = sum(
        term in lower for term in relation_terms
    )

    comparison_count = sum(
        term in lower for term in comparison_terms
    )

    # Diversity: proportion of unique terms across retrieved passages.
    total_passage_terms = sum(document_lengths)

    term_diversity = (
        len(retrieved_terms) / total_passage_terms
        if total_passage_terms else 0.0
    )

    features = [
        len(query_tokens),
        len(query_terms),
        len(question.get("candidate_passage_ids", [])),
        len(ranking[:5]),

        max(overlaps),
        float(np.mean(overlaps)),
        float(np.std(overlaps)),
        weighted_overlap,
        query_coverage,
        idf_coverage,

        max(title_overlaps),
        float(np.mean(title_overlaps)),

        float(np.mean(document_lengths))
        if document_lengths else 0.0,

        float(np.std(document_lengths))
        if document_lengths else 0.0,

        term_diversity,
        redundancy,

        len(entity_words),
        entity_coverage,

        relation_count,
        comparison_count,

        int(method == "vector"),
        int(method == "graph")
    ]

    return np.asarray([features], dtype=np.float64)


def make_classifier(y, balanced=True):
    if len(set(y)) < 2:
        return DummyClassifier(strategy="most_frequent")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if balanced else None,
            random_state=42
        ))
    ])


def positive_probability(model, features, positive_class=1):
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if positive_class not in classes:
        return 0.0

    return float(probabilities[classes.index(positive_class)])


def fit_classifier(X, y, balanced=True):
    model = make_classifier(y, balanced=balanced)
    model.fit(X, y)
    return model

def fit_utility_regressor(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0))
    ])
    model.fit(X, y)
    return model

def load_retrieval_runtime():
    import torch
    from sentence_transformers import SentenceTransformer

    passages = load_json(VECTOR_DIR / "passages.json")
    embeddings = np.load(VECTOR_DIR / "embeddings.npy").astype("float32")

    if len(passages) != len(embeddings):
        raise ValueError("Vector passages and embeddings do not align.")

    passage_indices = {
        p["id"]: i for i, p in enumerate(passages)
    }

    graph = load_json(GRAPH_FILE)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    return passages, embeddings, passage_indices, graph, model


def vector_rank(question, embeddings, passage_indices, model):
    candidates = question["candidate_passage_ids"]

    indices = [
        passage_indices[pid]
        for pid in candidates
        if pid in passage_indices
    ]

    if not indices:
        return []

    query = (
        "Represent this sentence for searching relevant passages: "
        + question["question"]
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")

    scores = embeddings[indices] @ query_embedding
    order = np.argsort(scores)[::-1][:5]

    return [
        # Map back to the original passage ID.
        next(
            pid for pid in candidates
            if passage_indices.get(pid) == indices[pos]
        )
        for pos in order
    ]


def graph_rank(question, graph, corpus):
    from src.graph.graph_pipeline import retrieve

    results = retrieve(
        question["question"], question, graph, corpus
    )

    return [r["passage_id"] for r in results[:5]]


def rrf_rank(vector_ids, graph_ids, k=60):
    scores = {}

    for ranking in (vector_ids, graph_ids):
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)[:5]


def retrieval_metrics(ranking, relevant):
    relevant = set(relevant)
    hits = relevant.intersection(ranking[:5])

    recall = len(hits) / len(relevant) if relevant else 0.0
    mrr = next(
        (1.0 / rank for rank, pid in enumerate(ranking[:5], 1)
         if pid in relevant),
        0.0
    )
    sufficient = int(
        bool(relevant) and relevant.issubset(set(ranking[:5]))
    )

    return recall, mrr, sufficient
