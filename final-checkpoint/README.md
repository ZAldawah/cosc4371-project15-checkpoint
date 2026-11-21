# COSC 4371 Project 15: Final Checkpoint

Zainulabdeen Aldawah - Fall 2026  
Repository: [cosc4371-project15-checkpoint](https://github.com/ZAldawah/cosc4371-project15-checkpoint)

This directory contains the final prompt-injection detection comparison. The primary variable remains the **test dataset**: table rows are Safe-Guard, deepset, and Gandalf; columns are TF-IDF/logistic regression, fine-tuned DistilBERT, and the pretrained ProtectAI detector. Recall is the primary metric, Safe-Guard is the reference, and the change is reference recall minus external-dataset recall. The decision boundaries are recorded in `audit/design.json`.

The frozen split contains 6,400 training and 1,591 validation rows, all from Safe-Guard, plus 3,715 test rows: 2,059 Safe-Guard, 656 deepset, and 1,000 Gandalf. Gandalf labels are presumed positive from collection provenance; its standalone evaluation supports detection rate, not false-positive rate.

## Reproduce

Run commands from this directory in a Python 3.12 environment. The recorded GPU environment uses Windows, Python 3.12.7, and PyTorch 2.9.1+cu126. Install the matching CUDA wheel before the remaining pinned dependencies:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python run_final.py
```

For CPU execution, replace the first command with:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

The model scripts select CUDA when available and otherwise use CPU, which is slower. These wheel sources follow the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v291).

`run_final.py` downloads missing dataset files into `data/`, recomputes the data audit, runs the classifiers and zero-shot comparison, assembles scores, runs the checker, builds the PDF, and verifies artifacts. Downloads use pinned dataset/model revisions; every dataset file is checked against its SHA-256. Internet access and space for model downloads are needed. Raw prompts and local weights are not redistributed with the public artifacts.

An existing raw-data cache can be supplied with `python run_final.py --data-dir path/to/cache`. Fine-tuned weights default to `.models/distilbert/`. Use `python run_final.py --reuse-distilbert` to reuse them: the transformer script validates the checkpoint checksum and training-metadata sidecar, including revision, split, seed, and training configuration. For a custom checkpoint location, call `run_transformers.py --model-dir path/to/models` directly; the orchestrator uses the default location.

## Methods and script order

1. `verify_data.py` reproduces exact/near-text groups and the original split, verifies input hashes and boundary overlaps, and records normalization controls.
2. `run_classical.py` fits word unigram/bigram TF-IDF plus logistic regression on training rows only. Each calibrated classifier uses the lowest clean-validation probability threshold with validation FPR at most 0.05. Predictions are positive when score is at least the threshold. It also accepts the fixed normalizer using validation data before test scoring.
3. `run_transformers.py` trains DistilBERT for three epochs, maximum 512 tokens, batch size 8 with two-step accumulation (effective batch 16). ProtectAI receives no local fine-tuning but uses its own probabilities on the same validation rows for calibration. Its original training overlap is unresolved. Thresholds stay fixed across test datasets and transformed views.
4. `run_zeroshot.py` implements Week 10 description matching with Qwen3-Embedding-0.6B instead of the lecture's 8B model. Fixed class descriptions, normalized embeddings, and cosine-margin threshold zero require no labeled training or calibration. This is separate from the primary table.
5. `assemble_results.py`, `build_final_report.py`, and `verify_final.py` generate and check the deliverables.

All eight supplemental views preserve original IDs and labels: clean, zero-width, homoglyph, Base64, and the normalized version of each. `scores_obfuscation.csv` contains four evaluation pools per classifier: the three original datasets and Gandalf plus all 1,409 held-out Safe-Guard benign rows. That mixed pool is artificial and not verified style matching. Normalization tests cover known reversible encodings; they do not demonstrate general attack prevention.

## Outputs and checks

| File or directory | Purpose |
|---|---|
| `Aldawah_COSC4371_Final_Checkpoint.pdf`, `figures/` | Report and figure |
| `split.csv`, `scores.csv`, `errors.csv` | Required primary checker inputs |
| `checker_output.txt` | Full primary CLI output, included in the report |
| `scores_obfuscation.csv`, `scores_zeroshot.csv` | Supplemental predictions |
| `checker_obfuscation_output.txt` | Checker-helper metrics, differences, and warnings |
| `checker_zeroshot_output.txt` | Supplemental zero-shot CLI output |
| `results/` | Per-model probabilities, validation outputs, and summaries |
| `audit/` | Design, hashes, timing/environment records, data checks, and qualitative annotations |

Recheck the primary files independently:

```powershell
python check_final_checkpoint.py --split split.csv --scores scores.csv --errors errors.csv --metric recall --reference safeguard --direction lower
```

Error sampling uses seed 4371, sorted IDs, and 10 false positives plus 10 false negatives from deepset/TF-IDF, or all available errors of a type. Preserved category annotations and rules are in `audit/error_annotations.csv` and `audit/error_category_rules.json`; assembly verifies the sampled IDs. The separate 150-record label review does not change source labels.

GPU training is seeded but permits nondeterministic-operation warnings; byte-identical retraining is not claimed. Exact-text inference caching is label-blind, so restored-view timing includes cache reuse. Checker warnings require interpretation even when its exit code is zero.

## AI disclosure

AI was used in the organization, formatting, and structuring of this document.
