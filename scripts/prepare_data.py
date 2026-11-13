"""Assemble the locally stored Shakespeare works for from-scratch training."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "shakespeare"
OUTPUT = ROOT / "data" / "processed" / "shakespeare_corpus.txt"
MANIFEST = ROOT / "data" / "processed" / "manifest.json"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")

def main() -> None:
    files = sorted(SOURCE.rglob("*.txt"))
    if not files:
        raise SystemExit("No Shakespeare source files found under data/shakespeare.")
    parts: list[str] = []
    manifest = []
    total_words = 0
    for path in files:
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue
        words = len(WORD_RE.findall(text))
        total_words += words
        parts.append(text)
        manifest.append({
            "path": path.relative_to(ROOT).as_posix(),
            "characters": len(text),
            "words": words,
            "category": path.parent.name,
        })
    # Include the canonical FAI source as a labelled training document.
    # This is not the missing subject archive; do not fabricate or infer that data.
    fai_source = ROOT / "docs" / "FAI_SOURCE_DYING_EMBER.txt"
    if fai_source.is_file():
        source_text = fai_source.read_text(encoding="utf-8").strip()
        if source_text:
            source_words = len(WORD_RE.findall(source_text))
            total_words += source_words
            parts.append("[Project Dying Ember canonical source]\n" + source_text)
            manifest.append({
                "path": fai_source.relative_to(ROOT).as_posix(),
                "characters": len(source_text),
                "words": source_words,
                "category": "project-source",
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "corpus": "Shakespeare collected works",
        "files": manifest,
        "file_count": len(manifest),
        "word_count": total_words,
        "character_count": sum(item["characters"] for item in manifest),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Assembled {len(manifest)} local work files · {total_words:,} words · {OUTPUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
