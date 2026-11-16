import json
import numpy as np
import torch
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("data/processed/corpus.jsonl")
INDEX_DIR = Path("indexes/vector")

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 4

INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

passages = load_jsonl(CORPUS_PATH)

if not passages:
    raise ValueError("Corpus is empty.")

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)
print("Loading:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME, device=device)

documents = [
    p["title"] + ". " + p["text"]
    for p in passages
]

print("Generating embeddings:", len(documents))

embeddings = model.encode(
    documents,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

embeddings = np.asarray(embeddings, dtype=np.float32)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

np.save(INDEX_DIR / "embeddings.npy", embeddings)

with (INDEX_DIR / "passages.json").open("w", encoding="utf-8") as f:
    json.dump(passages, f, ensure_ascii=False)

print("\nVECTOR INDEX BUILT")
print("Passages:", index.ntotal)
print("Dimensions:", dimension)
print("Index:", INDEX_DIR / "index.faiss")
print("Embedding shape:", embeddings.shape)
