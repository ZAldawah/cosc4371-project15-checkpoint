"""Strict zero-shot description matching, adapted from Week 10 Lecture 1.

Run after the fixed split exists: python run_zeroshot.py --data-dir /path/to/raw
Only test text is encoded; no project labels train or calibrate this model.
The course used Qwen3-Embedding-8B; this local run uses its 0.6B sibling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = "Determine whether the text is a prompt injection attempt or a benign request."
DESCRIPTIONS = {
    "benign": (
        "Benign request. A normal question, instruction, or piece of text for an assistant "
        "to process that does not attempt to override the assistant's governing instructions "
        "or redirect it away from its assigned task."
    ),
    "injection": (
        "Prompt injection. A message that tries to make an assistant ignore, override or "
        "reveal its hidden instructions, adopt an unauthorized role, or follow commands "
        "embedded in untrusted content instead of its assigned task."
    ),
}
HERE = Path(__file__).resolve().parent


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_configuration(args):
    """Write every modeling choice before reading test text or computing predictions."""
    from final_common import write_json
    config = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "course_reference": "Week 10 Lecture 1, pages 18 and 33-35; demo scene 7",
        "adaptation": "same description-matching method, 0.6B model rather than course 8B",
        "instruction": INSTRUCTION,
        "query_template": "Instruct: {instruction}\nQuery:{text}",
        "class_descriptions": DESCRIPTIONS,
        "description_instruction_prefix": False,
        "representation": "last unmasked token; L2-normalized full 1024-dimensional embedding",
        "score": "cosine(prompt,injection_description)-cosine(prompt,benign_description)",
        "threshold": 0.0,
        "tie_rule": "score >= 0 is positive",
        "threshold_selection": "fixed nearest-description rule; no labeled calibration",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "training_rows_used": 0,
        "validation_rows_used": 0,
        "project_label_use": "evaluation only after scores have been computed",
        "description_selection": "one fixed pair written before inference; no prompt search",
        "test_views": ["clean"],
        "source_file_sha256": sha256_file(__file__),
        "split_sha256": sha256_file(HERE / "split.csv"),
    }
    write_json(HERE / "audit" / "zeroshot_freeze.json", config)
    return config


def download_model():
    from huggingface_hub import snapshot_download
    from final_common import write_json
    started = time.perf_counter()
    path = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        max_workers=1,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt", "model.safetensors"],
    )
    write_json(HERE / "audit" / "zeroshot_download.json", {
        "model_id": MODEL_ID, "revision": REVISION,
        "download_or_cache_lookup_seconds": time.perf_counter() - started,
    })
    print(f"Pinned model cache ready: {MODEL_ID}@{REVISION}", flush=True)
    return path


def last_token_pool(hidden, attention_mask):
    import torch
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from final_common import load_data, write_json

    config = freeze_configuration(args)
    model_path = download_model()
    total_start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(4371)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=False, use_safetensors=True, local_files_only=True,
    ).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model_load_seconds = time.perf_counter() - load_start

    kept = load_data(args.data_dir)
    test = kept[kept.split == "test"].sort_values("row_id").reset_index(drop=True)
    assert test.row_id.is_unique
    query_texts = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in test.text]
    token_lengths = np.array([len(ids) for ids in tokenizer(
        query_texts, add_special_tokens=True, truncation=False,
        return_attention_mask=False,
    )["input_ids"]])
    truncation = {}
    for dataset, group in test.groupby("dataset", sort=True):
        lengths = token_lengths[group.index.to_numpy()]
        truncation[dataset] = {
            "rows": len(group),
            "truncated_rows": int((lengths > args.max_length).sum()),
            "max_untruncated_tokens": int(lengths.max()),
            "token_count_includes_instruction": True,
        }

    def encode(texts):
        vectors = []
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start:start + args.batch_size], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                output = model(**batch)
                pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
                vectors.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
            if len(texts) > 100 and (start // args.batch_size + 1) % 50 == 0:
                print(f"Zero-shot encoded {min(start+args.batch_size,len(texts))}/{len(texts)} rows", flush=True)
        return torch.cat(vectors, dim=0)

    inference_start = time.perf_counter()
    prototypes = encode([DESCRIPTIONS["benign"], DESCRIPTIONS["injection"]])
    embeddings = encode(query_texts)
    similarity = embeddings @ prototypes.T
    scores = (similarity[:, 1] - similarity[:, 0]).numpy().astype(float)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    assert len(scores) == len(test) and np.isfinite(scores).all()

    # Labels are copied for evaluation only after text-only inference is complete.
    result = test[["row_id", "dataset", "label"]].copy()
    result.insert(2, "view", "clean")
    result["score"] = scores
    result["threshold"] = 0.0
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / "zeroshot_scores.csv"
    result.to_csv(out, index=False, float_format="%.17g")
    import pandas as pd
    reread = pd.read_csv(out, float_precision="round_trip")
    assert np.array_equal(reread.score.to_numpy(), scores)
    assert set(reread.row_id) == set(test.row_id)

    report = dict(config)
    report.update({
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "huggingface-hub", "numpy", "pandas")},
        "model_load_seconds": model_load_seconds,
        "inference_seconds_including_class_descriptions": inference_seconds,
        "total_seconds_excluding_download": time.perf_counter() - total_start,
        "training_seconds": 0.0,
        "rows": len(test),
        "rows_by_dataset": {str(k): int(v) for k, v in test.groupby("dataset").size().items()},
        "truncation": truncation,
        "embedding_dimension": int(embeddings.shape[1]),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
        "pretraining_overlap_counts": "unknown: full pretrained corpus unavailable",
        "project_fine_tuning_overlap_counts": "not applicable: no project training or validation used",
        "interpretation_limit": "description wording, smaller model, truncation and unknown pretraining exposure limit comparisons; this is not a causal estimate of synthetic-data effects",
        "output_sha256": sha256_file(out),
    })
    write_json(HERE / "audit" / "zeroshot_results.json", report)
    print(f"Zero-shot completed: {len(test)} rows, inference {inference_seconds:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.max_length < 32 or args.batch_size < 1:
        parser.error("max-length must be >=32 and batch-size must be positive")
    if args.download_only:
        freeze_configuration(args)
        download_model()
    else:
        run(args)


if __name__ == "__main__":
    main()
