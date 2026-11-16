
import json

DATA = "data/processed/"

with open(DATA + "questions_dev.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

with open(DATA + "qrels_dev.jsonl", encoding="utf-8") as f:
    qrels = {
        item["question_id"]: item["supporting_passage_ids"]
        for item in map(json.loads, f)
    }

with open(DATA + "corpus.jsonl", encoding="utf-8") as f:
    corpus = {
        item["id"]: item
        for item in map(json.loads, f)
    }

for q in questions[:5]:
    print("\n" + "=" * 80)
    print("QUESTION:", q["question"])
    print("ANSWER:", q["answer"])

    gold_ids = qrels.get(q["id"], [])

    for pid in gold_ids:
        p = corpus.get(pid)

        print("\nGOLD PASSAGE:", pid)

        if p:
            print("TITLE:", p["title"])
            print("TEXT:", p["text"][:1200])
        else:
            print("ERROR: Passage missing from corpus")
