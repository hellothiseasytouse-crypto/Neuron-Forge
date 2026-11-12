"""Combine UTF-8 .txt files into one training corpus."""
import argparse
from pathlib import Path


def collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() == ".txt" else []
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    return sorted(p for p in source.rglob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="a .txt file or directory of .txt files")
    parser.add_argument("--output", default=Path("data/processed/corpus.txt"), type=Path)
    args = parser.parse_args()
    files = collect(args.input)
    if not files:
        raise SystemExit("No .txt files found in the input path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as output:
        for path in files:
            content = path.read_text(encoding="utf-8", errors="strict").strip()
            if content:
                output.write(content + "\n\n")
                written += len(content)
    if written == 0:
        raise SystemExit("The selected text files contained no usable text")
    print(f"Combined {len(files)} files ({written:,} characters) into {args.output}")


if __name__ == "__main__":
    main()
