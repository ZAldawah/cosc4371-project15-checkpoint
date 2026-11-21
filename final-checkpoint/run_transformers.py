"""Train and evaluate the fixed transformer comparisons for Project 15.

Raw prompt strings are classification inputs only. Model revisions are pinned,
remote Python code is disabled, and only safetensors weights are loaded.
Run from this directory: python run_transformers.py --models distilbert protectai
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from final_common import load_data, choose_threshold, transform_text

HERE = Path(__file__).resolve().parent
SEED = 4371
VIEWS = ("clean", "zero_width", "homoglyph", "base64", "normalized_zero_width",
         "normalized_homoglyph", "normalized_base64", "normalized_clean")
SOURCES = {
    "distilbert": ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"),
    "protectai": ("protectai/deberta-v3-base-prompt-injection-v2", "90c9989b1a342275dd0d1a95aad283c04e075671"),
}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_normalization_freeze():
    frozen = json.loads((HERE / "audit" / "normalization_freeze.json").read_text(encoding="utf-8"))
    source_hash = file_sha256(HERE / "final_common.py")
    if frozen.get("accepted") is not True or frozen.get("source_sha256") != source_hash:
        raise ValueError("Run the validation-only normalization freeze for the current final_common.py before transformer scoring")
    return source_hash


def expected_training(args, rows):
    update_steps = math.ceil(math.ceil(rows / args.train_batch_size) / args.gradient_accumulation) * args.epochs
    return {
        "seed": SEED, "rows": rows, "epochs": args.epochs,
        "batch_size": args.train_batch_size, "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation,
        "max_length": args.max_length, "optimizer": "torch.optim.AdamW", "learning_rate": args.learning_rate,
        "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
        "scheduler": "linear decay with 10% warmup", "warmup_steps": int(0.1 * update_steps),
        "optimizer_steps": update_steps, "gradient_norm_limit": 1.0,
        "selection_rule": "fixed final epoch; no test tuning or best-test checkpoint selection",
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def environment(device):
    names = ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
             "numpy", "pandas", "sentencepiece", "protobuf", "scikit-learn", "pyarrow")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in names},
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": "warn_only=True; seeded; cuDNN benchmark and TF32 disabled",
        "precision": "CUDA float16 autocast; logits converted to float32 before softmax" if device.type == "cuda" else "float32",
    }


class TokenRows(Dataset):
    def __init__(self, encoding, labels):
        self.encoding, self.labels = encoding, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encoding.items()}, "labels": int(self.labels[i])}


def train_distilbert(model, tokenizer, train, device, args, audit):
    start = time.perf_counter()
    encoded = tokenizer(train.text.tolist(), truncation=True, max_length=args.max_length, padding=False)
    dataset = TokenRows(encoded, train.label.to_numpy())
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                        generator=generator, num_workers=0,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    audit["training_tokenize_seconds"] = time.perf_counter() - start
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01,
                                 betas=(0.9, 0.999), eps=1e-8)
    update_steps = math.ceil(len(loader) / args.gradient_accumulation) * args.epochs
    warmup_steps = int(0.1 * update_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    audit["training"] = expected_training(args, len(train))
    epochs = []
    microsteps = 0
    sync(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        sum_loss, seen = 0.0, 0
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            count = len(batch["labels"])
            sum_loss += float(loss.detach().cpu()) * count
            seen += count
            scaler.scale(loss / args.gradient_accumulation).backward()
            if (b + 1) % args.gradient_accumulation == 0 or b + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            microsteps += 1
            if microsteps == 20:
                sync(device)
                elapsed = time.perf_counter() - start
                audit["pilot"] = {
                    "measured_microbatches": 20, "rows": seen, "seconds": elapsed,
                    "rows_per_second": seen / elapsed,
                    "projected_full_training_seconds": elapsed / 20 * len(loader) * args.epochs,
                    "scope": "First 20 training microbatches, including warmup, optimizer updates and loss transfer; continued same run without restart",
                }
                save_json(HERE / "audit" / "transformer_distilbert.json", audit)
                print(f"DistilBERT pilot: {elapsed:.2f}s / 20 batches; projected full fit {audit['pilot']['projected_full_training_seconds']/60:.2f} min", flush=True)
            if (b + 1) % 100 == 0:
                print(f"DistilBERT epoch {epoch+1}/{args.epochs}: batch {b+1}/{len(loader)}; loss {sum_loss/seen:.5f}", flush=True)
        sync(device)
        epochs.append({"epoch": epoch + 1, "mean_train_loss": sum_loss / seen,
                       "seconds": time.perf_counter() - epoch_start})
    sync(device)
    audit["training_seconds"] = time.perf_counter() - start
    audit["epoch_log"] = epochs
    audit["optimizer_steps_completed"] = int(scheduler.last_epoch)
    return model


class Scorer:
    """Memoize exact input texts only; cached prediction never uses a label."""
    def __init__(self, model, tokenizer, device, max_length, batch_size, positive_index=1):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_length, self.batch_size, self.positive_index = max_length, batch_size, positive_index
        self.scores, self.lengths = {}, {}

    def predict(self, texts):
        verify_normalization_freeze()
        unique = list(dict.fromkeys(t for t in texts if t not in self.scores))
        start = time.perf_counter()
        if unique:
            full = self.tokenizer(unique, truncation=False, padding=False)
            self.lengths.update({t: len(ids) for t, ids in zip(unique, full["input_ids"])})
            del full
            unique.sort(key=lambda t: (min(self.lengths[t], self.max_length), t))
            self.model.eval()
            with torch.inference_mode():
                for i in range(0, len(unique), self.batch_size):
                    texts_batch = unique[i:i+self.batch_size]
                    encoded = self.tokenizer(texts_batch, truncation=True, max_length=self.max_length,
                                             padding=True, pad_to_multiple_of=8, return_tensors="pt")
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                        logits = self.model(**encoded).logits
                    probabilities = torch.softmax(logits.float(), dim=-1)[:, self.positive_index].cpu().numpy()
                    self.scores.update(zip(texts_batch, map(float, probabilities)))
                    if (i // self.batch_size + 1) % 100 == 0:
                        print(f"  inference {min(i+self.batch_size,len(unique))}/{len(unique)} new texts", flush=True)
        sync(self.device)
        values = np.array([self.scores[t] for t in texts], dtype=np.float64)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        return values, {"seconds": time.perf_counter() - start, "rows": len(texts),
                        "new_unique_texts": len(unique), "cache_hits": len(texts)-len(unique),
                        "truncated_rows": sum(self.lengths[t] > self.max_length for t in texts),
                        "maximum_untruncated_tokens": max(self.lengths[t] for t in texts)}


def run_model(name, data, device, args):
    seed_all()
    normalization_source_sha256 = verify_normalization_freeze()
    train, val, test = [data[data.split == split].sort_values("row_id").reset_index(drop=True)
                        for split in ("train", "validation", "test")]
    model_id, revision = SOURCES[name]
    audit = {"model": name, "model_id": model_id, "revision": revision,
             "source_url": f"https://huggingface.co/{model_id}/tree/{revision}",
             "environment": environment(device), "seed": SEED,
             "split_sha256": hashlib.sha256((HERE / "split.csv").read_bytes()).hexdigest(),
             "normalization_source_sha256": normalization_source_sha256,
             "same_training_set": name == "distilbert",
             "training_exposure_note": "Same project Safe-Guard training set" if name == "distilbert" else "Frozen pretrained reference; publisher training overlap with these benchmarks is unresolved",
             "max_length": args.max_length, "prediction_cache": "Exact text only, label blind, reused across views", "status": "running"}
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    checkpoint = args.model_dir / "distilbert"
    if name == "distilbert" and args.reuse_distilbert:
        previous = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
        expected = {"model_id": model_id, "revision": revision, "split_sha256": audit["split_sha256"],
                    "seed": SEED, "training": expected_training(args, len(train)),
                    "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors")}
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"Saved DistilBERT checkpoint metadata mismatch: {key}; retrain instead of reusing")
        if previous.get("training_completed") is not True:
            raise ValueError("Saved DistilBERT checkpoint has no completed-training record")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, trust_remote_code=False,
                                                                   use_safetensors=True)
        for key in ("training", "training_tokenize_seconds", "training_seconds", "pilot", "epoch_log", "optimizer_steps_completed"):
            audit[key] = previous[key]
        audit["checkpoint_sha256"] = previous["checkpoint_sha256"]
        audit["reused_saved_checkpoint"] = True
    else:
        kwargs = {"num_labels": 2, "id2label": {0: "SAFE", 1: "INJECTION"}, "label2id": {"SAFE": 0, "INJECTION": 1}} if name == "distilbert" else {}
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
                    trust_remote_code=False, use_safetensors=True, **kwargs)
    model.to(device)
    audit["download_and_load_seconds"] = time.perf_counter() - start
    if name == "distilbert" and not audit.get("reused_saved_checkpoint"):
        model = train_distilbert(model, tokenizer, train, device, args, audit)
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        audit["checkpoint_sha256"] = file_sha256(checkpoint / "model.safetensors")
        training_metadata = {**audit, "training_completed": True}
        save_json(checkpoint / "training_metadata.json", training_metadata)
        save_json(HERE / "audit" / "transformer_distilbert.json", audit)
    elif name != "distilbert":
        audit["training_seconds"] = 0.0
    audit["model_label_mapping"] = model.config.id2label
    positive_index = int(model.config.label2id["INJECTION"])
    scorer = Scorer(model, tokenizer, device, args.max_length, args.inference_batch_size, positive_index)
    val_scores, audit["validation_scoring"] = scorer.predict(val.text.tolist())
    threshold = float(choose_threshold(val.label.to_numpy(), val_scores))
    validation = val[["row_id", "dataset", "label", "group_id"]].copy()
    validation["score"], validation["threshold"] = val_scores, threshold
    validation.to_csv(HERE / "results" / f"{name}_validation_scores.csv", index=False, float_format="%.17g")
    audit["threshold"], audit["threshold_repr"] = threshold, repr(threshold)
    audit["threshold_rule"] = "lowest unique validation probability (plus nextafter maximum) with FPR <=0.05; predictions score>=threshold; frozen across test datasets and views"
    neg = val.label.to_numpy() == 0
    pos = ~neg
    audit["validation"] = {"rows": len(val), "negative_rows": int(neg.sum()), "positive_rows": int(pos.sum()),
                           "false_positives": int((val_scores[neg] >= threshold).sum()),
                           "fpr": float((val_scores[neg] >= threshold).mean()),
                           "recall": float((val_scores[pos] >= threshold).mean())}
    save_json(HERE / "audit" / f"transformer_{name}.json", audit)
    print(f"{name}: validation threshold={threshold!r}, FPR={audit['validation']['fpr']:.5f}", flush=True)
    frames, view_audits = [], {}
    for view in VIEWS:
        print(f"{name}: scoring {view}", flush=True)
        texts = [transform_text(t, view) for t in test.text]
        scores, details = scorer.predict(texts)
        frame = test[["row_id", "dataset", "label"]].copy()
        frame["view"], frame["score"], frame["threshold"] = view, scores, threshold
        frame = frame[["row_id", "dataset", "view", "label", "score", "threshold"]]
        frames.append(frame)
        details["datasets"] = {}
        for dataset_name in sorted(test.dataset.unique()):
            mask = test.dataset.to_numpy() == dataset_name
            selected = [t for t, keep in zip(texts, mask) if keep]
            labels = test.label.to_numpy()[mask]
            predictions = scores[mask] >= threshold
            details["datasets"][dataset_name] = {
                "rows": len(selected), "positive_rows": int(labels.sum()),
                "truncated_rows": sum(scorer.lengths[t] > args.max_length for t in selected),
                "recall": float(predictions[labels == 1].mean()) if (labels == 1).any() else None,
                "fpr": float(predictions[labels == 0].mean()) if (labels == 0).any() else None,
            }
        view_audits[view] = details
        pd.concat(frames, ignore_index=True).to_csv(HERE / "results" / f"{name}_scores.csv", index=False, float_format="%.17g")
        audit["views"] = view_audits
        save_json(HERE / "audit" / f"transformer_{name}.json", audit)
        print(f"{name}: {view} completed in {details['seconds']:.1f}s", flush=True)
    audit["status"] = "complete"
    audit["test_scoring_seconds"] = sum(v["seconds"] for v in view_audits.values())
    audit["timing_interpretation"] = "Wall-clock scoring time includes tokenization and exact-text cache hits; restored views can reuse prior inputs and are not independent full model-inference runs"
    audit["total_unique_prediction_inputs"] = len(scorer.scores)
    save_json(HERE / "audit" / f"transformer_{name}.json", audit)
    del scorer, model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=tuple(SOURCES), nargs="+", default=list(SOURCES))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=HERE / ".models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--reuse-distilbert", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_all()
    device = torch.device(args.device)
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir)
    assert data.row_id.is_unique
    print(f"Running on {environment(device)['gpu'] or device}; train/validation/test counts {data.split.value_counts().to_dict()}", flush=True)
    for name in args.models:
        run_model(name, data, device, args)


if __name__ == "__main__":
    main()
