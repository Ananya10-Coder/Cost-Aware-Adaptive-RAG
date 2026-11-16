import json

for filename in [
    "data/processed/questions_dev.jsonl",
    "data/processed/qrels_dev.jsonl",
]:
    print("\n---", filename, "---")
    with open(filename, encoding="utf-8") as f:
        for line in list(f)[:2]:
            print(json.dumps(json.loads(line), indent=2)[:2500])

with open("indexes/graph/pilot_graph.json", encoding="utf-8") as f:
    graph = json.load(f)

print("\n--- GRAPH STRUCTURE ---")
print("Node example:", json.dumps(graph["nodes"][0], indent=2)[:1500])
print("Edge example:", json.dumps(graph["edges"][0], indent=2)[:1500])