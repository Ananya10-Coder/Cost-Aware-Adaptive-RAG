import json
import shutil
from pathlib import Path

DATA = Path("data/processed")
RESULTS = Path("results")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize(value):
    value = str(value)
    if value.startswith("musique:"):
        return value[len("musique:"):]
    return value

# Original 500 question IDs from the existing experiment
with open(RESULTS / "router_training_data.json", encoding="utf-8") as f:
    router_rows = json.load(f)

old_qids = {normalize(r["question_id"]) for r in router_rows}

if len(old_qids) != 500:
    raise ValueError(
        f"Expected 500 original questions, found {len(old_qids)}"
    )

questions_path = DATA / "questions_dev.jsonl"
qrels_path = DATA / "qrels_dev.jsonl"
corpus_path = DATA / "corpus.jsonl"

questions = read_jsonl(questions_path)
qrels = read_jsonl(qrels_path)
corpus = read_jsonl(corpus_path)

# Select only the original questions
old_questions = [
    q for q in questions
    if normalize(q["id"]) in old_qids
]

if len(old_questions) != 500:
    raise ValueError(
        f"Could not recover all 500 questions. Found {len(old_questions)}. "
        "No files have been overwritten."
    )

# Restore original, unprefixed question IDs
for q in old_questions:
    q["id"] = normalize(q["id"])
    q["candidate_passage_ids"] = [
        normalize(pid) for pid in q["candidate_passage_ids"]
    ]

old_qrels = [
    q for q in qrels
    if normalize(q["question_id"]) in old_qids
]

for q in old_qrels:
    q["question_id"] = normalize(q["question_id"])
    q["supporting_passage_ids"] = [
        normalize(pid) for pid in q["supporting_passage_ids"]
    ]

# Keep only passages referenced by the original questions
needed_passages = set()

for q in old_questions:
    needed_passages.update(q["candidate_passage_ids"])

for q in old_qrels:
    needed_passages.update(q["supporting_passage_ids"])

old_corpus = [
    p for p in corpus
    if normalize(p["id"]) in needed_passages
]

for p in old_corpus:
    p["id"] = normalize(p["id"])

corpus_ids = {p["id"] for p in old_corpus}

missing = needed_passages - corpus_ids
if missing:
    raise ValueError(
        f"{len(missing)} required passages are missing. "
        "No files have been overwritten. Example: {list(missing)[:5]}"
    )

# Back up the combined dataset before replacing anything
for path in [questions_path, qrels_path, corpus_path]:
    backup = path.with_suffix(path.suffix + ".combined_backup")
    if not backup.exists():
        shutil.copy2(path, backup)

# Write restored MuSiQue-only files
write_jsonl(questions_path, old_questions)
write_jsonl(qrels_path, old_qrels)
write_jsonl(corpus_path, old_corpus)

print("\nRESTORATION COMPLETE")
print("Questions:", len(old_questions))
print("Qrels:", len(old_qrels))
print("Corpus passages:", len(old_corpus))
print("Original question IDs restored.")
print("MultiHop-RAG excluded.")
print("Combined dataset backups saved.")
