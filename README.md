**Final checkpoint:** [Report, code, and results](final-checkpoint/) | [Final report PDF](final-checkpoint/Aldawah_COSC4371_Final_Checkpoint.pdf)

---

# COSC 4371 Project 15: midsemester checkpoint

Zainulabdeen Aldawah — prompt-injection detection across datasets.

The approved comparison varies the **test dataset** while training stays on Safe-Guard. This checkpoint implements one TF-IDF/logistic-regression baseline. The final project also calls for the second classifier, zero-width/homoglyph/base64 obfuscations, normalization recovery, and the ProtectAI comparison; these are not implemented by this checkpoint script.

## Reproduce

Use Python 3.12. From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_checkpoint.py
```

On other systems, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` for the last two commands. The run downloads seven pinned Parquet files into `data/`, verifies their SHA-256 hashes, audits and splits them, trains the baseline, selects the threshold on validation, scores test rows, and runs the supplied checker. The near-duplicate audit covers all loaded rows and can take several minutes.

To reuse an existing download cache, preserve the dataset subdirectories and paths listed in `data_manifest.json`:

```powershell
.\.venv\Scripts\python.exe run_checkpoint.py --data-dir "C:\path\to\raw"
```

Run the supplied checker separately with the exact validation-selected threshold:

```powershell
.\.venv\Scripts\python.exe check_checkpoint.py --split split.csv --scores scores.csv --threshold "0.14544270512937266"
```

The command is also recorded in `audit/results.json` under `checker.command`; the threshold is in `baseline.threshold_repr`. Keep the quotation marks in PowerShell to preserve every digit. Do not replace it with a rounded value. Checker output is saved in `checker_output.txt`.

## Data and split

The source URLs, full revision IDs, file paths, and checksums are in `data_manifest.json`:

| Dataset | Role | Revision |
| --- | --- | --- |
| [Safe-Guard](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | Retained publisher training rows supply train/validation; retained publisher test rows remain test | `a3a877d608f37b7d20d9945671902df895ecdb46` |
| [deepset](https://huggingface.co/datasets/deepset/prompt-injections) | All released splits pooled into one external test set | `4f61ecb038e9c3fb77e21034b22511b523772cdd` |
| [Gandalf](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | All released splits pooled into one external test set | `04737b65e90a6794ec227012e4a255a7def6344b` |

Safe-Guard and deepset use their supplied binary labels. Gandalf has no binary target; the script assigns 1 to its presumed injection attempts and reports detection rate (the checker's recall), not verified ground-truth accuracy. Gandalf's `similarity` field is excluded from the model. Raw datasets are downloaded from their sources and are not redistributed in this repository; their redistribution terms have not been fully established here.

Exact duplicates mean identical original text, excluding the label. Near duplicates use Unicode NFKC, case folding, collapsed whitespace, and character 5-gram set Jaccard similarity of at least 0.85. Strings shorter than five characters use the whole normalized string as one token. A lossless prefix/length filter generates candidates, followed by exact set comparison; connected components define text families, including transitive chains.

The script quarantines invalid/missing rows, exact texts with conflicting supplied labels, and every exact cross-dataset overlap. It removes Safe-Guard development rows in any exact/near family touching a publisher or external test row. Retained development families are split with the first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4371)`, approximately 80% training and 20% validation. No released Safe-Guard test row enters development. User/session identifiers are unavailable, so text families cannot establish independence of authors or sessions. Repeats within a partition remain; the supplied checker's row-bootstrap intervals do not account for family dependence.

## Model and outputs

The pipeline fits word unigram/bigram TF-IDF on training text only (`min_df=2`, `max_features=100000`, sublinear term frequency, L2 normalization), followed by logistic regression (`C=1`, L2 penalty, `liblinear`, `max_iter=2000`, seed 4371). Audit normalization is not used as model preprocessing. The threshold is the lowest validation-score candidate with validation false-positive rate at most 5%; predictions are positive when `score >= threshold`. Test probabilities are computed only after the threshold is fixed. Primary comparison: recall at this fixed threshold across the three test sets.

| File | Contents |
| --- | --- |
| `split.csv` | Every retained row: globally unique `row_id`, `dataset`, `split` (`train`, `validation`, `test`), binary `label`; extra audit columns are included |
| `scores.csv` | Every test row: matching `row_id`, `test_set`, `label`, and positive-class probability `score` |
| `validation_scores.csv` | Validation IDs, labels, family IDs, and probabilities used to select the threshold |
| `audit/results.json` | Input audits, cleaning/split counts, leakage checks, metrics and intervals, exact threshold, timings, hardware, Python/package versions, and checker command |
| `audit/duplicate_groups.csv`, `audit/near_duplicate_edges.csv`, `audit/removed_rows.csv` | Hash-based family/cleaning evidence without prompt text |
| `checker_output.txt` | Supplied checker's split counts, metrics with 95% intervals, and warnings |
| `figures/deepset_confusion_matrix.png` | Baseline confusion matrix on the retained deepset test set |

`check_checkpoint.py` is the instructor-supplied checker. Re-running the pipeline replaces its generated artifacts. Timings vary by hardware; the downloaded bytes must match the manifest. Consult the report for interpretation and explanations of checker warnings. Public repository: <https://github.com/ZAldawah/cosc4371-project15-checkpoint>.

A fresh source-only copy downloaded all seven files and reproduced `split.csv`, `scores.csv`, validation scores, cleaning/group evidence, and checker output byte for byte. See `audit/fresh_run_verification.json`.

The included four-page PDF is the report for this recorded run. `prepare_report.py` and `build_report.py` preserve its source. PDF regeneration additionally needs ReportLab and the Windows Times New Roman fonts; it is optional and is not part of the baseline reproduction command.

## AI assistance

[5] OpenAI. ChatGPT/Codex, October 29, 2026. https://chatgpt.com/. AI assistance was used for formatting and verification purposes.
