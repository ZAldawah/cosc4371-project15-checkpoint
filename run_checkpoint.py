"""Reproduce the COSC 4371 Project 15 baseline; raw prompts are data only.

Run: python run_checkpoint.py
Optional existing cache: --data-dir /path/to/proposal/audit/raw
No prompt is sent to an LLM or executed. Downloads are pinned and SHA-256 checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import check_checkpoint as checker

SEED = 4371
HERE = Path(__file__).resolve().parent
DATASET_NAMES = {
    "xTRam1/safe-guard-prompt-injection": "safeguard",
    "deepset/prompt-injections": "deepset",
    "Lakera/gandalf_ignore_instructions": "gandalf",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(text: str) -> str:
    """Audit-only normalization: Unicode NFKC, casefold, whitespace collapse."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def grams(text: str) -> set[str]:
    value = normalize(text)
    # For <5 characters use the entire string as one token, avoiding empty sets.
    return {value[i:i + 5] for i in range(len(value) - 4)} if len(value) >= 5 else {value}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def read_data(data_dir: Path):
    manifest = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    records, input_files = [], []
    for item in manifest["files"]:
        path = data_dir / item["local_path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading pinned {item['repository']} {item['remote_path']}", flush=True)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "COSC4371-reproducibility/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            if sha256_bytes(content) != item["sha256"]:
                raise ValueError(f"Downloaded SHA-256 mismatch: {item['local_path']}")
            path.write_bytes(content)
        actual_hash = sha256_bytes(path.read_bytes())
        if actual_hash != item["sha256"]:
            raise ValueError(f"Cached SHA-256 mismatch: {path}")
        source = pd.read_parquet(path)
        name = DATASET_NAMES[item["repository"]]
        source_split = Path(item["remote_path"]).name.split("-")[0]
        if "text" not in source.columns:
            raise ValueError(f"Missing text field: {path}")
        input_files.append({"dataset": name, "source_split": source_split,
                            "rows": len(source), "columns": list(source.columns),
                            "column_count": len(source.columns), "sha256": actual_hash,
                            "revision": item["revision"], "local_path": item["local_path"],
                            "missing_per_column": {str(k): int(v) for k, v in source.isna().sum().items()}})
        for i, row in source.iterrows():
            raw_label = row.get("label", np.nan)
            presumed = name == "gandalf"
            valid_label = presumed or (pd.notna(raw_label) and raw_label in (0, 1))
            raw_text = row["text"]
            valid_text = isinstance(raw_text, str) and bool(raw_text.strip())
            text = raw_text if isinstance(raw_text, str) else ""
            records.append({"row_id": f"{name}:{source_split}:{i}", "dataset": name,
                            "source_split": source_split, "text": text,
                            "label": 1 if presumed else int(raw_label) if valid_label else -1,
                            "label_kind": "presumed_positive" if presumed else "source_label",
                            "missing_target": bool(pd.isna(raw_label)), "valid_label": valid_label,
                            "missing_text": not valid_text, "text_sha256": sha256_bytes(text.encode("utf-8"))})
    df = pd.DataFrame(records).sort_values(["dataset", "source_split", "row_id"]).reset_index(drop=True)
    assert df.row_id.is_unique
    audit = {}
    for dataset, group in df.groupby("dataset", sort=True):
        counts = group.text.value_counts()
        cols = next(x["columns"] for x in input_files if x["dataset"] == dataset)
        audit[dataset] = {
            "rows": len(group), "columns": cols, "column_count": len(cols),
            "positive_rows": int((group.label == 1).sum()), "negative_rows": int((group.label == 0).sum()),
            "positive_percent": 100 * float((group.label == 1).mean()),
            "negative_percent": 100 * float((group.label == 0).mean()),
            "label_interpretation": "presumed positives; binary source target unavailable" if dataset == "gandalf" else "supplied binary source label",
            "missing_source_target": int(group.missing_target.sum()),
            "invalid_supplied_label_rows": int((~group.valid_label).sum()),
            "missing_or_blank_text_rows": int(group.missing_text.sum()),
            "exact_duplicate_excess": int((counts - 1).sum()),
            "exact_duplicate_groups": int((counts > 1).sum()),
            "exact_duplicate_involved_rows": int(counts[counts > 1].sum()),
            "exact_duplicate_pairs": int(sum(int(n) * (int(n) - 1) // 2 for n in counts)),
            "duplicate_definition": "identical original text, label excluded; excess = rows - unique text strings",
            "repeated_entity_counts": "unavailable: no user, session, host, or author identifier field",
            "time_range": "unavailable: no per-row timestamp field",
        }
    return df, {"files": input_files, "datasets": audit, "total_rows": len(df)}


def duplicate_audit(df: pd.DataFrame, output_dir: Path):
    """Exact all-pairs Jaccard join using lossless prefix and length filters.

    Globally order tokens by document frequency then lexical value. For size n,
    the first n-ceil(.85*n)+1 tokens are the Jaccard prefix. Every qualifying
    pair shares a prefix token and has smaller_size >= ceil(.85*larger_size).
    We verify candidates using exact set intersection and integer comparison:
    20*intersection >= 17*union. No hashing approximation or sampling occurs.
    """
    start = time.perf_counter()
    unique_texts = sorted(df.text.unique())
    text_ids = {text: i for i, text in enumerate(unique_texts)}
    unique_to_rows = defaultdict(list)
    for i, text in enumerate(df.text):
        unique_to_rows[text_ids[text]].append(i)
    print(f"Full near-duplicate audit: {len(df)} rows; {len(unique_texts)} distinct raw texts", flush=True)
    token_sets = [grams(text) for text in unique_texts]
    freq = Counter(token for tokens in token_sets for token in tokens)
    token_rank = {token: rank for rank, token in enumerate(sorted(freq, key=lambda t: (freq[t], t)))}
    token_sets = [set(token_rank[token] for token in tokens) for tokens in token_sets]
    sizes = [len(tokens) for tokens in token_sets]
    del freq, token_rank
    uf = UnionFind(len(unique_texts))
    inverted = defaultdict(list)
    edges = []
    candidate_checks = 0
    primary_pairs, primary_vertices = 0, set()
    primary_rows_in_pair = set()
    all_row_pairs = 0
    order = sorted(range(len(unique_texts)), key=lambda i: (sizes[i], i))
    for done, i in enumerate(order):
        tokens, n = token_sets[i], sizes[i]
        minimum_size = (17 * n + 19) // 20
        prefix = sorted(tokens)[:n - minimum_size + 1]
        candidates = set()
        for token in prefix:
            candidates.update(j for j in inverted[token] if sizes[j] >= minimum_size)
        for j in sorted(candidates):
            candidate_checks += 1
            intersection = len(tokens & token_sets[j])
            union = n + sizes[j] - intersection
            if 20 * intersection >= 17 * union:
                uf.union(i, j)
                rows_i, rows_j = unique_to_rows[i], unique_to_rows[j]
                pi = [r for r in rows_i if df.iloc[r].dataset == "safeguard"]
                pj = [r for r in rows_j if df.iloc[r].dataset == "safeguard"]
                if pi and pj:
                    primary_pairs += len(pi) * len(pj)
                    primary_vertices.update((i, j))
                    primary_rows_in_pair.update(pi + pj)
                pairs = len(rows_i) * len(rows_j)
                all_row_pairs += pairs
                edges.append({"text_sha256_a": sha256_bytes(unique_texts[i].encode()),
                              "text_sha256_b": sha256_bytes(unique_texts[j].encode()),
                              "intersection": intersection, "union": union,
                              "jaccard": intersection / union, "row_pairs": pairs,
                              "safeguard_row_pairs": len(pi) * len(pj)})
        for token in prefix:
            inverted[token].append(i)
        if (done + 1) % 2000 == 0:
            print(f"  {done + 1}/{len(order)} unique texts; {len(edges)} near edges; {time.perf_counter()-start:.1f}s", flush=True)
    df["group_id"] = [f"g{uf.find(text_ids[text]):05d}" for text in df.text]
    pd.DataFrame(edges, columns=["text_sha256_a", "text_sha256_b", "intersection", "union", "jaccard", "row_pairs", "safeguard_row_pairs"]).to_csv(output_dir / "near_duplicate_edges.csv", index=False)
    df[["row_id", "dataset", "source_split", "label", "text_sha256", "group_id"]].to_csv(output_dir / "duplicate_groups.csv", index=False)
    group_sizes = df.groupby("group_id").size()
    sg_group_sizes = df[df.dataset == "safeguard"].groupby("group_id").size()
    report = {"rule": "NFKC, casefold, whitespace collapse; distinct character 5-gram sets; Jaccard >= 0.85; strings shorter than 5 use their whole string as one token",
              "scope": "all 11958 input rows, every dataset; no subsampling",
              "method": "lossless globally-frequency-ordered prefix filtering plus length filtering and exact integer set-intersection verification",
              "threshold": 0.85, "candidate_pairs_verified": candidate_checks,
              "distinct_raw_texts": len(unique_texts), "near_unique_text_pairs": len(edges),
              "near_row_pairs_excluding_identical_raw_text": all_row_pairs,
              "primary_near_row_pairs_excluding_identical_raw_text": primary_pairs,
              "primary_distinct_raw_texts_in_near_pairs": len(primary_vertices),
              "primary_rows_in_near_pairs": len(primary_rows_in_pair),
              "all_exact_or_near_components": int(len(group_sizes)),
              "all_non_singleton_components": int((group_sizes > 1).sum()),
              "primary_exact_or_near_non_singleton_components": int((sg_group_sizes > 1).sum()),
              "primary_exact_or_near_component_excess": int((sg_group_sizes - 1).sum()),
              "elapsed_seconds": time.perf_counter() - start,
              "transitivity": "Connected components include chains whose endpoints need not directly meet the threshold."}
    print(f"Full audit complete: {len(edges)} near text pairs in {report['elapsed_seconds']:.2f}s", flush=True)
    return df, report


def clean_split(df: pd.DataFrame, output_dir: Path):
    supplied = df[df.label_kind == "source_label"]
    conflict_keys = set(supplied.groupby("text").label.nunique().loc[lambda x: x > 1].index)
    cross_keys = set(df.groupby("text").dataset.nunique().loc[lambda x: x > 1].index)
    anchors = df[(df.dataset != "safeguard") | (df.source_split == "test")]
    anchor_groups = set(anchors.group_id)
    development = (df.dataset == "safeguard") & (df.source_split == "train")
    reasons = []
    for row in df.itertuples():
        flags = []
        if not row.valid_label or row.missing_text:
            flags.append("invalid_or_missing_target_or_text")
        if row.text in conflict_keys:
            flags.append("exact_label_conflict")
        if row.text in cross_keys:
            flags.append("cross_dataset_exact_overlap")
        if row.dataset == "safeguard" and row.source_split == "train" and row.group_id in anchor_groups:
            flags.append("development_group_touches_test")
        reasons.append(flags)
    df["removal_reasons"] = [";".join(r) for r in reasons]
    removed = df[df.removal_reasons != ""].copy()
    removed["primary_removal_reason"] = removed.removal_reasons.str.split(";").str[0]
    removed[["row_id", "dataset", "source_split", "label", "text_sha256", "group_id", "removal_reasons", "primary_removal_reason"]].to_csv(output_dir / "removed_rows.csv", index=False)
    kept = df[df.removal_reasons == ""].copy()
    kept["split"] = "test"
    dev = kept[(kept.dataset == "safeguard") & (kept.source_split == "train")]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_pos, val_pos = next(splitter.split(dev.text, dev.label, dev.group_id))
    kept.loc[dev.iloc[train_pos].index, "split"] = "train"
    kept.loc[dev.iloc[val_pos].index, "split"] = "validation"
    kept["test_set"] = np.where(kept.split == "test", kept.dataset, "")
    kept = kept.sort_values(["dataset", "split", "row_id"]).reset_index(drop=True)
    assert (kept[kept.split != "test"].dataset == "safeguard").all()
    assert (kept[(kept.dataset == "safeguard") & (kept.source_split == "test")].split == "test").all()
    checks = {}
    train = kept[kept.split == "train"]
    val = kept[kept.split == "validation"]
    for name, test in kept[kept.split == "test"].groupby("test_set"):
        checks[f"train_rows_exact_match_{name}_test"] = int(train.text.isin(set(test.text)).sum())
        checks[f"train_groups_overlap_{name}_test"] = len(set(train.group_id) & set(test.group_id))
        checks[f"validation_groups_overlap_{name}_test"] = len(set(val.group_id) & set(test.group_id))
    checks["train_rows_exact_match_validation"] = int(train.text.isin(set(val.text)).sum())
    checks["train_groups_overlap_validation"] = len(set(train.group_id) & set(val.group_id))
    assert all(n == 0 for n in checks.values()), checks
    boundary = []
    for name, rows in anchors.groupby(["dataset", "source_split"]):
        touched = df[development & df.group_id.isin(set(rows.group_id))]
        exact = touched.text.isin(set(rows.text))
        boundary.append({"protected_test_dataset": name[0], "protected_source_split": name[1],
                         "development_rows_in_touching_components": len(touched),
                         "direct_exact_development_rows": int(exact.sum()),
                         "near_or_transitive_only_development_rows": int((~exact).sum())})
    report = {"input_rows": len(df), "kept_rows": len(kept), "removed_rows": len(removed),
              "moved_test_rows_into_development": 0,
              "conflicting_exact_text_keys": len(conflict_keys), "cross_dataset_exact_text_keys": len(cross_keys),
              "removal_flag_counts_overlapping": {flag: sum(flag in r for r in reasons) for flag in sorted({r for flags in reasons for r in flags})},
              "exclusive_primary_removal_counts": {str(k): int(v) for k, v in removed.primary_removal_reason.value_counts().items()},
              "removed_by_dataset_source_split": removed.groupby(["dataset", "source_split"]).size().rename("rows").reset_index().to_dict("records"),
              "protected_boundaries_overlapping": boundary,
              "rule": "quarantine missing/invalid rows, exact supplied-label conflicts, and every cross-dataset exact-overlap row; remove SafeGuard development rows in any full-input exact/near component touching released SafeGuard test or any external row; keep publisher SafeGuard test in test; pool all external source splits by dataset",
              "split_rule": "first fold of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4371) on retained SafeGuard released train; exact/near connected component is the group",
              "retained_development_rows": len(dev), "train_rows": len(train), "validation_rows": len(val),
              "validation_fraction_of_retained_development": len(val) / len(dev),
              "groups": {sp: int(g.group_id.nunique()) for sp, g in kept.groupby("split")},
              "leakage_checks": checks,
              "same_partition_repetitions": "Within-partition exact/near repeats are retained; checker intervals resample rows and do not model template-group dependence.",
              "related_entity_limit": "Entity IDs are unavailable; only measured exact/near text families can be grouped."}
    return kept, report


def hardware():
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "logical_cpus": os.cpu_count(), "training_device": "CPU only", "fit_thread_limit": 1}
    if os.name == "nt":
        try:
            script = "$p=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; [pscustomobject]@{name=$p.Name; cores=$p.NumberOfCores; logical_processors=$p.NumberOfLogicalProcessors; ram_bytes=$m.TotalPhysicalMemory}|ConvertTo-Json -Compress"
            result["windows_hardware"] = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", script], timeout=30, text=True))
        except Exception as exc:
            result["hardware_query_note"] = str(exc)
    return result


def train_score(kept: pd.DataFrame, out: Path):
    train, val, test = [kept[kept.split == sp] for sp in ("train", "validation", "test")]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  max_features=100000, sublinear_tf=True, norm="l2")),
        ("logistic", LogisticRegression(C=1.0, solver="liblinear", penalty="l2",
                                        class_weight=None, max_iter=2000, tol=1e-4, random_state=SEED)),
    ])
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(train.text, train.label)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    validation_scores = model.predict_proba(val.text)[:, 1]
    validation_score_seconds = time.perf_counter() - start
    candidates = np.append(np.unique(validation_scores), np.nextafter(validation_scores.max(), np.inf))
    negative_scores = validation_scores[val.label.to_numpy() == 0]
    assert len(negative_scores) > 0
    threshold = next(float(t) for t in candidates if np.mean(negative_scores >= t) <= 0.05)
    val_pred = validation_scores >= threshold
    validation_output = val[["row_id", "label", "group_id"]].copy()
    validation_output["score"] = validation_scores
    validation_output.to_csv(out / "validation_scores.csv", index=False, float_format="%.17g")
    # Exactly one predict_proba call for the pooled test rows, after threshold freezes.
    start = time.perf_counter()
    test_scores = model.predict_proba(test.text)[:, 1]
    test_score_seconds = time.perf_counter() - start
    scored = test[["row_id", "test_set", "label"]].copy()
    scored["score"] = test_scores
    scored.to_csv(out / "scores.csv", index=False, float_format="%.17g")
    # Read back with the checker's parser before calculating its published values.
    scored = pd.read_csv(out / "scores.csv")
    rng = np.random.default_rng(0)
    metrics = {}
    for name, g in scored.groupby("test_set", sort=False):
        y, s = g.label.to_numpy(), g.score.to_numpy()
        pred = (s >= threshold).astype(int)
        values = checker.all_metrics(y, s, pred)
        cis = checker.bootstrap(y, s, pred, rng, 1000)
        metrics[name] = {"test_rows": len(g), "positive_rows": int(y.sum()), "negative_rows": int(len(y)-y.sum()),
                         "positive_percent": 100 * float(y.mean()), "random_scorer_AP": float(y.mean()) if y.min() != y.max() else None,
                         "metrics": {key: {"value": None if value is None else float(value), "ci95": cis[key]} for key, value in values.items()},
                         "f1": float(f1_score(y, pred)) if y.min() != y.max() else None,
                         "confusion_matrix_true_rows_0_1_pred_cols_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                         "one_class_report_only": "detection rate (checker recall on presumed positives)" if name == "gandalf" else None}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(metrics["deepset"]["confusion_matrix_true_rows_0_1_pred_cols_0_1"])
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    ax.imshow(cm, cmap="Blues", vmin=0)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=17,
                color="white" if value > cm.max() / 2 else "#11263e")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Injection"],
           yticklabels=["Legitimate", "Injection"], xlabel="Predicted class", ylabel="Source label",
           title="TF-IDF + logistic regression: deepset")
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "deepset_confusion_matrix.png", dpi=220)
    plt.close(fig)
    return {"fit_seconds": fit_seconds, "validation_score_seconds": validation_score_seconds,
            "test_score_seconds": test_score_seconds, "threshold": threshold, "threshold_repr": repr(threshold),
            "threshold_rule": "lowest candidate with validation FPR <= 0.05; candidates are unique validation probabilities plus nextafter(max_probability, +inf); positive iff score >= threshold",
            "validation": {"rows": len(val), "positive_rows": int(val.label.sum()), "negative_rows": len(negative_scores),
                           "false_positives": int(val_pred[val.label.to_numpy() == 0].sum()),
                           "false_positive_rate": float(val_pred[val.label.to_numpy() == 0].mean()),
                           "recall": float(val_pred[val.label.to_numpy() == 1].mean())},
            "pipeline": {"preprocessing": "model receives raw text; TfidfVectorizer lowercase=True, default Unicode word token pattern (?u)\\b\\w\\w+\\b; no accent stripping or audit normalization",
                         "representation": "TF-IDF word unigrams/bigrams; min_df=2; max_features=100000; sublinear_tf=True; smooth_idf=True; norm=l2; vocabulary and IDF fitted on training only",
                         "classifier": "LogisticRegression C=1, penalty=l2, solver=liblinear, class_weight=None, tol=1e-4, max_iter=2000, random_state=4371",
                         "fitted_vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                         "solver_iterations": model.named_steps["logistic"].n_iter_.tolist(),
                         "predictors_excluded": ["dataset", "source_split", "label", "similarity", "row_id", "group_id"]},
            "test_prediction_calls": 1, "metrics": metrics, "bootstrap": {"resamples": 1000, "seed": 0, "method": "unstratified row bootstrap, percentile 2.5/97.5; exact supplied checker implementation and test-set order"},
            "primary_metric_widest_ci_half_width": max((m["metrics"]["recall"]["ci95"][1]-m["metrics"]["recall"]["ci95"][0])/2 for m in metrics.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data", help="existing raw directory or download cache (default: repository/data)")
    args = parser.parse_args()
    start = time.perf_counter()
    audit_dir = HERE / "audit"
    audit_dir.mkdir(exist_ok=True)
    df, audit = read_data(args.data_dir.resolve())
    df, near = duplicate_audit(df, audit_dir)
    kept, cleanup = clean_split(df, audit_dir)
    kept[["row_id", "dataset", "split", "label", "source_split", "label_kind", "group_id", "text_sha256"]].to_csv(HERE / "split.csv", index=False)
    print(f"Retained {len(kept)} rows; fitting fixed baseline", flush=True)
    result = train_score(kept, HERE)
    package_names = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "joblib", "threadpoolctl"]
    results = {"seed": SEED, "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in package_names},
               "hardware": hardware(), "input_audit": audit, "near_duplicates": near, "cleanup": cleanup,
               "split_counts": kept.groupby(["dataset", "split"]).label.agg(rows="size", positive="sum").assign(negative=lambda x: x.rows-x.positive).reset_index().to_dict("records"),
               "baseline": result, "checker_sha256": sha256_bytes((HERE / "check_checkpoint.py").read_bytes())}
    command = [sys.executable, str(HERE / "check_checkpoint.py"), "--split", str(HERE / "split.csv"), "--scores", str(HERE / "scores.csv"), "--threshold", result["threshold_repr"]]
    checked = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    (HERE / "checker_output.txt").write_text(checked.stdout + checked.stderr, encoding="utf-8")
    results["checker"] = {"returncode": checked.returncode, "command": 'python check_checkpoint.py --split split.csv --scores scores.csv --threshold "' + result["threshold_repr"] + '"',
                          "warnings": checked.stdout.split("Warnings\n--------\n")[-1].strip(),
                          "output_file": "checker_output.txt"}
    results["elapsed_total_seconds"] = time.perf_counter() - start
    (audit_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(checked.stdout, flush=True)
    if checked.returncode:
        raise RuntimeError(checked.stderr or checked.stdout)
    print(f"Completed in {results['elapsed_total_seconds']:.2f}s; fit {result['fit_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
