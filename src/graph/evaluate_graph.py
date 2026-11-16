import json
from pathlib import Path
from graph_retriever import retrieve

DATA = Path("data/processed")

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

questions = load_jsonl(DATA / "questions_dev.jsonl")[:5]
qrels = {
    x["question_id"]: set(x["supporting_passage_ids"])
    for x in load_jsonl(DATA / "qrels_dev.jsonl")
}

recalls = []
mrrs = []

for q in questions:
    results = retrieve(q["question"], top_k=5)

    retrieved = [r["passage_id"] for r in results]
    relevant = qrels[q["id"]]

    hits = set(retrieved) & relevant
    recall = len(hits) / len(relevant) if relevant else 0

    reciprocal_rank = 0
    for rank, pid in enumerate(retrieved, 1):
        if pid in relevant:
            reciprocal_rank = 1 / rank
            break

    recalls.append(recall)
    mrrs.append(reciprocal_rank)

    print("\nQuestion:", q["question"])
    print("Gold passages:", len(relevant))
    print("Retrieved:", len(retrieved))
    print("Recall@5:", round(recall, 3))
    print("MRR:", round(reciprocal_rank, 3))

print("\nGRAPH PILOT RESULTS")
print("Questions:", len(questions))
print("Mean Recall@5:", round(sum(recalls) / len(recalls), 4))
print("Mean MRR:", round(sum(mrrs) / len(mrrs), 4))
