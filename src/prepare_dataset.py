
import json
import hashlib
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

RAW = Path("data/raw")
OUT = Path("data/processed")

MUSIQUE_FILE = RAW / "musique_ans_v1.0_dev.jsonl"
MULTIHOP_Q = RAW / "multihop_rag/MultiHopRAG.json"
MULTIHOP_CORPUS = RAW / "multihop_rag/corpus.json"

N_MUSIQUE = 500
N_MULTIHOP = 500
SEED = 42


def make_id(prefix, *values):
    raw = "::".join(str(v or "").strip() for v in values)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_outputs():
    existing = [
        OUT / name for name in (
            "corpus.jsonl",
            "questions_dev.jsonl",
            "qrels_dev.jsonl"
        ) if (OUT / name).exists()
    ]

    if existing:
        folder = OUT / (
            "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        folder.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, folder / path.name)
        print("Previous data backed up to:", folder)


def prepare_musique(rng):
    with open(MUSIQUE_FILE, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    answerable = [x for x in raw if x.get("answerable", True)]

    if len(answerable) < N_MUSIQUE:
        raise ValueError("Not enough answerable MuSiQue questions.")

    selected = rng.sample(answerable, N_MUSIQUE)

    corpus, questions, qrels = {}, [], []

    for item in selected:
        candidates, supporting = [], []

        for p in item["paragraphs"]:
            title = p["title"].strip()
            text = p["paragraph_text"].strip()

            if not text:
                continue

            pid = make_id("musique", title, text)

            corpus[pid] = {
                "id": pid,
                "dataset": "musique",
                "title": title,
                "text": text
            }

            candidates.append(pid)

            if p.get("is_supporting", False):
                supporting.append(pid)

        candidates = list(dict.fromkeys(candidates))
        supporting = list(dict.fromkeys(supporting))

        if not candidates or not supporting:
            raise ValueError(f"Invalid MuSiQue question: {item['id']}")

        qid = f"musique:{item['id']}"

        questions.append({
            "id": qid,
            "dataset": "musique",
            "question": item["question"],
            "answer": item["answer"],
            "answer_aliases": item.get("answer_aliases", []),
            "candidate_passage_ids": candidates
        })

        qrels.append({
            "question_id": qid,
            "supporting_passage_ids": supporting
        })

    return corpus, questions, qrels, len(answerable)


def prepare_multihop(rng):
    raw_questions = load_json(MULTIHOP_Q)
    raw_articles = load_json(MULTIHOP_CORPUS)

    corpus = {}
    title_map = defaultdict(list)
    url_map = defaultdict(list)

    for article in raw_articles:
        title = str(article.get("title", "")).strip()
        body = str(article.get("body", "")).strip()
        url = str(article.get("url", "")).strip()

        if not title or not body:
            raise ValueError("MultiHop article missing title/body.")

        pid = make_id("multihop", url or title, body)

        corpus[pid] = {
            "id": pid,
            "dataset": "multihop_rag",
            "title": title,
            "text": body,
            "author": article.get("author") or "",
            "source": article.get("source") or "",
            "published_at": article.get("published_at") or "",
            "category": article.get("category") or "",
            "url": url
        }

        title_map[norm(title)].append(pid)

        if url:
            url_map[norm(url)].append(pid)

    # Exclude null queries: they have no gold evidence and are
    # evaluated separately, not as answerable Recall@5 questions.
    answerable = [
        x for x in raw_questions
        if x.get("question_type") != "null_query"
        and x.get("evidence_list")
    ]

    # Validate evidence mappings BEFORE sampling, so we can obtain
    # exactly 500 usable questions.
    valid = []

    for item in answerable:
        matched = []

        for evidence in item["evidence_list"]:
            url = str(evidence.get("url", "")).strip()
            title = str(evidence.get("title", "")).strip()

            ids = url_map.get(norm(url), []) if url else []

            if not ids and title:
                ids = title_map.get(norm(title), [])

            if not ids:
                matched = []
                break

            matched.extend(ids)

        matched = list(dict.fromkeys(matched))

        if matched:
            valid.append((item, matched))

    if len(valid) < N_MULTIHOP:
        raise ValueError(
            f"Only {len(valid)} MultiHop questions have mapped evidence; "
            f"need {N_MULTIHOP}. Check article matching."
        )

    # Proportional stratified sample by question type.
    groups = defaultdict(list)

    for item, ids in valid:
        groups[item.get("question_type", "unknown")].append((item, ids))

    for group in groups.values():
        rng.shuffle(group)

    total = len(valid)
    quotas = {
        key: int(N_MULTIHOP * len(group) / total)
        for key, group in groups.items()
    }

    remainder = N_MULTIHOP - sum(quotas.values())

    keys = sorted(
        groups,
        key=lambda k: (
            N_MULTIHOP * len(groups[k]) / total - quotas[k]
        ),
        reverse=True
    )

    for key in keys[:remainder]:
        quotas[key] += 1

    selected = []

    for key, group in groups.items():
        selected.extend(group[:quotas[key]])

    rng.shuffle(selected)

    questions, qrels = [], []
    all_article_ids = list(corpus.keys())

    for item, supporting in selected:
        qid = make_id(
            "multihop_q",
            item["query"],
            item.get("answer", "")
        )

        questions.append({
            "id": qid,
            "dataset": "multihop_rag",
            "question": item["query"],
            "answer": item.get("answer", ""),
            "answer_aliases": [],
            "question_type": item.get("question_type", "unknown"),
            "candidate_passage_ids": all_article_ids
        })

        qrels.append({
            "question_id": qid,
            "supporting_passage_ids": supporting
        })

    return corpus, questions, qrels, len(answerable)


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Preparing MuSiQue...")
    mc, mq, mr, m_available = prepare_musique(rng)

    print("Preparing MultiHop-RAG...")
    hc, hq, hr, h_available = prepare_multihop(rng)

    corpus = {**mc, **hc}
    questions = mq + hq
    qrels = mr + hr

    rng.shuffle(questions)
    rng.shuffle(qrels)

    # Validate IDs and evidence before overwriting old files.
    qids = [q["id"] for q in questions]
    rids = [q["question_id"] for q in qrels]

    assert len(questions) == 1000
    assert len(set(qids)) == 1000
    assert set(qids) == set(rids)

    corpus_ids = set(corpus)
    qrel_map = {
        q["question_id"]: set(q["supporting_passage_ids"])
        for q in qrels
    }

    for q in questions:
        candidates = set(q["candidate_passage_ids"])
        relevant = qrel_map[q["id"]]

        assert candidates.issubset(corpus_ids)
        assert relevant
        assert relevant.issubset(candidates)

    backup_outputs()

    write_jsonl(OUT / "corpus.jsonl", list(corpus.values()))
    write_jsonl(OUT / "questions_dev.jsonl", questions)
    write_jsonl(OUT / "qrels_dev.jsonl", qrels)

    print("\nDATASET PREPARATION COMPLETE")
    print("MuSiQue available:", m_available)
    print("MuSiQue selected:", len(mq))
    print("MultiHop available with evidence:", h_available)
    print("MultiHop selected:", len(hq))
    print("MultiHop types:", dict(Counter(
        q.get("question_type", "unknown") for q in hq
    )))
    print("Total questions:", len(questions))
    print("Total corpus documents/passages:", len(corpus))
    print("Dataset counts:", dict(Counter(q["dataset"] for q in questions)))
    print("Saved all three processed files.")
    print("\nRebuild vector embeddings and graph index before evaluation.")


if __name__ == "__main__":
    main()
