
import json
from collections import defaultdict

with open("indexes/graph/pilot_graph.json", encoding="utf-8") as f:
    graph = json.load(f)

with open("data/processed/questions_dev.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

with open("data/processed/qrels_dev.jsonl", encoding="utf-8") as f:
    qrels = {
        q["question_id"]: q["supporting_passage_ids"]
        for q in map(json.loads, f)
    }

nodes = {n["id"]: n for n in graph["nodes"]}

passage_nodes = defaultdict(list)
for node in graph["nodes"]:
    for pid in node.get("passage_ids", []):
        passage_nodes[pid].append(node["id"])

edges_by_node = defaultdict(list)
for edge in graph["edges"]:
    edges_by_node[edge["source"]].append(edge)
    edges_by_node[edge["target"]].append(edge)

for q in questions[:5]:
    print("\n" + "=" * 75)
    print("QUESTION:", q["question"])

    for pid in qrels.get(q["id"], []):
        print("\nGOLD PASSAGE:", pid)

        if pid not in passage_nodes:
            print("NOT REPRESENTED IN GRAPH")
            continue

        entity_nodes = passage_nodes[pid]
        print("ENTITY NODES:", entity_nodes[:20])

        for node_id in entity_nodes[:10]:
            print("\nNODE:", node_id)

            for edge in edges_by_node[node_id][:5]:
                print(
                    " ",
                    edge["source"],
                    "--",
                    edge["relation"],
                    "-->",
                    edge["target"],
                    "| passage:",
                    edge.get("passage_id")
                )
