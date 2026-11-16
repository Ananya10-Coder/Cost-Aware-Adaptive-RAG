import json

path = "data/raw/musique_ans_v1.0_dev.jsonl"

with open(path, "r", encoding="utf-8") as f:
    first = json.loads(next(f))

print("Available fields:")
print(first.keys())

print("\nQuestion:")
print(first.get("question"))

print("\nAnswer:")
print(first.get("answer"))

paragraphs = first.get("paragraphs", [])

print("\nNumber of paragraphs:", len(paragraphs))

if paragraphs:
    print("\nFirst paragraph fields:")
    print(paragraphs[0].keys())

    print("\nFirst paragraph:")
    print(paragraphs[0].get("paragraph_text", "")[:500])

    print("\nSupporting paragraph:")
    print(paragraphs[0].get("is_supporting", False))
