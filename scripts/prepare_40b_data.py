"""Stream licensed text datasets, train the 65k BPE tokenizer, and write token shards.

The resulting files are local training artifacts; do not commit large corpus files
or model checkpoints to Git. Every source must be reviewed for its license first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


def read_documents(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                yield item


def document_text(item: dict) -> str:
    parts = [item.get("title", ""), item.get("text", "")]
    return "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def stream_sources(config: dict, raw_path: Path, max_documents: int) -> tuple[int, list[dict]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with: pip install 'datasets>=3.0'") from exc

    sources = [source for source in config.get("sources", []) if source.get("enabled", False)]
    if not sources:
        raise ValueError("No enabled data sources were found in the source manifest.")

    document_count = 0
    seen_content: set[str] = set()
    source_reports: list[dict] = []
    with raw_path.open("w", encoding="utf-8") as output:
        for source in sources:
            dataset_id = source["dataset_id"]
            dataset_config = source.get("dataset_config")
            split = source.get("split", "train")
            kwargs = {"split": split, "streaming": True}
            if dataset_config:
                dataset = load_dataset(dataset_id, dataset_config, **kwargs)
            else:
                dataset = load_dataset(dataset_id, **kwargs)

            accepted = 0
            skipped = 0
            print(f"Streaming {dataset_id} ({dataset_config or 'default'}, split={split})", flush=True)
            for row in dataset:
                if max_documents and document_count >= max_documents:
                    break
                text = row.get(source.get("text_field", "text"))
                if not isinstance(text, str):
                    skipped += 1
                    continue
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < int(config.get("minimum_document_characters", 300)):
                    skipped += 1
                    continue
                # Exact normalized-content deduplication prevents repeated documents
                # from leaking across train/validation splits.
                digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                if digest in seen_content:
                    skipped += 1
                    continue
                seen_content.add(digest)
                title = row.get(source.get("title_field", "title"), "")
                url = row.get(source.get("url_field", "url"), "")
                record = {
                    "text": text,
                    "title": title if isinstance(title, str) else "",
                    "url": url if isinstance(url, str) else "",
                    "source": source["name"],
                    "dataset_id": dataset_id,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                document_count += 1
                if accepted % 10000 == 0:
                    print(f"  accepted {accepted:,} documents from {dataset_id}", flush=True)
            source_reports.append({
                "name": source["name"],
                "dataset_id": dataset_id,
                "dataset_config": dataset_config,
                "split": split,
                "license": source.get("license", "UNSPECIFIED - do not use until reviewed"),
                "attribution": source.get("attribution", ""),
                "dataset_card": source.get("dataset_card", ""),
                "accepted_documents": accepted,
                "skipped_documents": skipped,
                "deduplication": "exact normalized text SHA-256",
            })
            if max_documents and document_count >= max_documents:
                break
    return document_count, source_reports


def train_tokenizer(raw_path: Path, output_path: Path, vocab_size: int, sample_documents: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        max_token_length=64,
        show_progress=True,
    )

    def examples():
        for index, item in enumerate(read_documents(raw_path)):
            if sample_documents and index >= sample_documents:
                break
            yield document_text(item)

    print(f"Training tokenizer (vocabulary target {vocab_size:,})...", flush=True)
    tokenizer.train_from_iterator(examples(), trainer=trainer)
    actual = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual != vocab_size:
        raise RuntimeError(
            f"Tokenizer learned {actual:,} tokens, expected {vocab_size:,}. "
            "The corpus does not contain enough diverse text; add more reviewed sources."
        )
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved: {output_path}", flush=True)
    return tokenizer


def write_token_shards(
    raw_path: Path,
    tokenizer: Tokenizer,
    output_dir: Path,
    validation_percent: int,
) -> dict:
    train_path = output_dir / "train.bin"
    validation_path = output_dir / "validation.bin"
    train_tokens = 0
    validation_tokens = 0
    document_count = 0

    with train_path.open("wb") as train_file, validation_path.open("wb") as validation_file:
        for item in read_documents(raw_path):
            content = document_text(item)
            ids = tokenizer.encode(content).ids
            if not ids:
                continue
            if max(ids) >= 65536:
                raise ValueError("Tokenizer emitted an ID that does not fit in uint16 storage.")
            key = f"{item.get('source','')}|{item.get('url','')}|{item.get('title','')}"
            bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
            values = np.asarray(ids, dtype=np.uint16)
            if bucket < validation_percent:
                values.tofile(validation_file)
                validation_tokens += int(values.size)
            else:
                values.tofile(train_file)
                train_tokens += int(values.size)
            document_count += 1
            if document_count % 10000 == 0:
                print(
                    f"Encoded {document_count:,} documents | train {train_tokens:,} tokens "
                    f"| validation {validation_tokens:,} tokens",
                    flush=True,
                )

    if train_tokens < 2049 or validation_tokens < 2049:
        raise RuntimeError(
            "The tokenized corpus is too small for the 2,048-token context. "
            "Add more reviewed datasets before training."
        )
    return {
        "documents_tokenized": document_count,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "storage_dtype": "uint16",
        "train_file": train_path.name,
        "validation_file": validation_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/40b-sources.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/40b"))
    parser.add_argument("--max-documents", type=int, default=0, help="0 means all documents in enabled sources")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_documents < 0:
        parser.error("--max-documents must be zero or greater")
    if not args.config.is_file():
        raise FileNotFoundError(f"Source manifest not found: {args.config}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "corpus.jsonl"
    partial_path = args.output_dir / "corpus.jsonl.partial"
    tokenizer_path = args.output_dir / "tokenizer.json"
    train_path = args.output_dir / "train.bin"
    validation_path = args.output_dir / "validation.bin"
    manifest_path = args.output_dir / "dataset-manifest.json"
    existing = [path for path in (raw_path, partial_path, tokenizer_path, train_path, validation_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "40B data artifacts already exist. Use --overwrite to rebuild them: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    print("Preparing 40B training data from approved sources.", flush=True)
    print("This downloads data to the machine running this script; it does not add the corpus to GitHub.", flush=True)
    document_count, source_reports = stream_sources(config, partial_path, args.max_documents)
    if document_count == 0:
        raise RuntimeError("No usable documents were downloaded.")
    partial_path.replace(raw_path)

    tokenizer = train_tokenizer(
        raw_path,
        tokenizer_path,
        int(config.get("tokenizer_vocab_size", 65536)),
        int(config.get("tokenizer_documents", 250000)),
    )
    shard_report = write_token_shards(
        raw_path,
        tokenizer,
        args.output_dir,
        int(config.get("validation_percent", 5)),
    )
    manifest = {
        "project": "Neuron Forge",
        "model_target": "40B-class",
        "status": "prepared_training_data",
        "documents_downloaded": document_count,
        "tokenizer_vocabulary": tokenizer.get_vocab_size(with_added_tokens=True),
        "sources": source_reports,
        **shard_report,
        "attribution_note": (
            "This manifest records declared source licenses and attribution. "
            "Verify every dataset card and upstream license before training, publishing weights, "
            "or redistributing generated outputs. Dataset terms may differ by source."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Prepared data directory: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
