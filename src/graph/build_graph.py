import json
import re
from pathlib import Path
from itertools import combinations

import networkx as nx
import spacy

DATA = Path("data/processed")
OUT = Path("indexes/graph")

PILOT_QUESTIONS = 5

OUT.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

corpus = {
    p["id"]: p for p in load_jsonl(DATA / "corpus.jsonl")
}
questions = load_jsonl(DATA / "questions_dev.jsonl")

# Use the actual candidate passages for the first five questions.
selected_ids = set()

for q in questions[:PILOT_QUESTIONS]:
    selected_ids.update(q["candidate_passage_ids"])

passages = [corpus[pid] for pid in selected_ids if pid in corpus]

G = nx.MultiDiGraph()

ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"
}

def add_node(name, passage_id, label="CONCEPT"):
    name = normalize(name)

    if not name or len(name) < 2 or len(name) > 100:
        return None

    if name not in G:
        G.add_node(name, label=label, passage_ids=[])

    if passage_id not in G.nodes[name]["passage_ids"]:
        G.nodes[name]["passage_ids"].append(passage_id)

    return name

def add_edge(subject, relation, obj, passage_id):
    if not subject or not obj or subject == obj:
        return

    G.add_edge(
        subject,
        obj,
        relation=relation,
        passage_id=passage_id
    )

for i, passage in enumerate(passages, 1):
    pid = passage["id"]
    text = passage["title"] + ". " + passage["text"]

    doc = nlp(text)

    # Extract named entities as graph nodes.
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            add_node(ent.text, pid, ent.label_)

    # Extract noun phrases as additional graph nodes.
    for chunk in doc.noun_chunks:
        add_node(chunk.text, pid)

    # Extract subject -> verb -> object relationships.
    for sent in doc.sents:
        entities_in_sentence = [
            ent for ent in sent.ents
            if ent.label_ in ENTITY_LABELS
        ]

        for verb in sent:
            if verb.pos_ != "VERB":
                continue

            subjects = [
                child for child in verb.children
                if child.dep_ in ("nsubj", "nsubjpass", "csubj")
            ]

            objects = [
                child for child in verb.children
                if child.dep_ in ("obj", "dobj", "attr", "oprd")
            ]

            for prep in verb.children:
                if prep.dep_ == "prep":
                    objects.extend(
                        child for child in prep.children
                        if child.dep_ == "pobj"
                    )

            for subj in subjects:
                for obj in objects:
                    subject_name = add_node(
                        subj.text, pid, "CONCEPT"
                    )
                    object_name = add_node(
                        obj.text, pid, "CONCEPT"
                    )

                    relation = verb.lemma_.lower()

                    add_edge(
                        subject_name,
                        relation,
                        object_name,
                        pid
                    )

        # Preserve entity co-occurrence relationships where
        # a direct subject-object relation was not extracted.
        for a, b in combinations(entities_in_sentence, 2):
            left = add_node(a.text, pid, a.label_)
            right = add_node(b.text, pid, b.label_)

            if left and right and left != right:
                if not G.has_edge(left, right):
                    add_edge(left, "co_occurs", right, pid)

    if i % 20 == 0:
        print(f"Processed {i}/{len(passages)} passages")

# Save a portable JSON representation.
graph_data = {
    "nodes": [
        {"id": node, **attrs}
        for node, attrs in G.nodes(data=True)
    ],
    "edges": [
        {
            "source": source,
            "target": target,
            **attrs
        }
        for source, target, attrs in G.edges(data=True)
    ]
}

with (OUT / "pilot_graph.json").open("w", encoding="utf-8") as f:
    json.dump(graph_data, f, ensure_ascii=False)

# JSON is our primary graph storage format.
# GraphML export is optional and requires converting list attributes.
nx.write_graphml(
    nx.Graph(
        (u, v, {
            key: str(value) if isinstance(value, (list, dict, set)) else value
            for key, value in attrs.items()
        })
        for u, v, attrs in G.edges(data=True)
    ),
    OUT / "pilot_graph.graphml"
)

print("\nGRAPH PILOT COMPLETE")
print("Questions:", min(PILOT_QUESTIONS, len(questions)))
print("Passages:", len(passages))
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Graph connected components:",
      nx.number_weakly_connected_components(G))
print("Saved:", OUT / "pilot_graph.json")
