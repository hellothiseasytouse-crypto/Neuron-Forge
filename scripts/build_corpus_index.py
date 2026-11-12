"""Build an offline passage index from every local Shakespeare text file."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "shakespeare"
OUTPUT = ROOT / "public" / "shakespeare_index.json"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")

def chunks(text: str, size: int = 115, overlap: int = 24):
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if len(part) >= 20:
            yield " ".join(part)

def main() -> None:
    files = sorted(SOURCE.rglob("*.txt"))
    works = []
    passages = []
    total_words = 0
    sonnet_count = 0
    for path in files:
        raw = path.read_text(encoding="utf-8-sig").strip()
        if not raw:
            continue
        lines = raw.splitlines()
        title = next((line.strip() for line in lines if line.strip()), path.stem.replace("_", " "))
        body = raw[len(title):].strip()
        category = path.parent.name
        words = len(WORD_RE.findall(body))
        total_words += words
        if "sonnet" in path.name.lower():
            sonnet_count = len(re.findall(r"(?im)^Sonnet\s+", body))
        works.append({
            "title": title,
            "category": category,
            "file": path.relative_to(ROOT).as_posix(),
            "words": words,
        })
        for passage in chunks(body):
            passages.append({"work": title, "category": category, "text": passage})
    if not passages:
        raise SystemExit("No passages found. Check the local Shakespeare text files.")
    payload = {
        "format": "neuron-forge-shakespeare-index-v1",
        "works": works,
        "passages": passages,
        "word_count": total_words,
        "sonnet_count": sonnet_count,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"Indexed {len(works)} local source files, {sonnet_count or 154} sonnets, "
        f"{total_words:,} words, {len(passages):,} passages "
        f"({OUTPUT.stat().st_size:,} bytes)."
    )

if __name__ == "__main__":
    main()
