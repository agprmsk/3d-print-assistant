import json
import pathlib

IN_PATH = pathlib.Path("data/raw/3dtoday_raw.jsonl")
OUT_PATH = pathlib.Path("data/processed.jsonl")

def normalize():
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            doc = {
                "id": item["url"],
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "source_url": item["url"],
                "category": item.get("category", ""),
                "tags": item.get("tags", []),
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    normalize()
