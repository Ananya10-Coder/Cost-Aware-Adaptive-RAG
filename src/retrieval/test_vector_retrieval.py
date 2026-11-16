import json
import faiss
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_DIR = Path("indexes/vector")

index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

with (INDEX_DIR / "passages.json").open(encoding="utf-8") as f:
    passages = json.load(f)

with open("data/processed/questions_dev.jsonl", encoding="utf-8") as f:
    question = json.loads(next(f))

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

query = (
    "Represent this sentence for searching relevant passages: "
    + question["question"]
)

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True
)

scores, indices = index.search(query_embedding, 5)

print("\nQUESTION:", question["question"])
print("EXPECTED ANSWER:", question["answer"])
print("\nTOP 5 RETRIEVED PASSAGES")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    p = passages[idx]

    print(f"\n{rank}. Score: {score:.4f}")
    print("Title:", p["title"])
    print("Passage:", p["text"][:500])
