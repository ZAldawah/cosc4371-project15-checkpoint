"""Build report content from the recorded baseline results (no manual metric entry)."""
import json
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'repository' if (ROOT / 'repository').is_dir() else ROOT
r = json.loads((REPO / 'audit/results.json').read_text())
b = r['baseline']; a = r['input_audit']['datasets']; c = r['cleanup']; near = r['near_duplicates']
m = b['metrics']; threshold = b['threshold_repr']; blocks = []
config_path = ROOT / 'publication.json'
publication = json.loads(config_path.read_text()) if config_path.exists() else {}
url = publication.get('url')
def p(text, kind='body'): blocks.append({'kind':kind,'text':text})
def table(rows, widths, font=9): blocks.append({'kind':'table','rows':rows,'widths':widths,'font':font})
def page(): blocks.append({'kind':'break'})
def metric(ds, name='recall'): return m[ds]['metrics'][name]['value']
def interval(ds, name):
    x = m[ds]['metrics'][name]
    if ds == 'gandalf' and name != 'recall': return 'one class'
    if x['value'] is None: return 'undefined'
    lo,hi=x['ci95']
    return f"{x['value']:.4f}<br/>[{lo:.4f}, {hi:.4f}]"
def link(url, label): return f'<link href="{escape(url,quote=True)}" color="#163b6b">{label}</link>'
names = {'safeguard':'Safe-Guard', 'deepset':'deepset', 'gandalf':'Gandalf'}

p('Midsemester Checkpoint: Prompt-Injection Detection Across Datasets','title')
p('Zainulabdeen Aldawah | COSC 4371 | Project 15 | October 29, 2026','small')
if url: p('Code repository: '+link(url,url),'small')
else: p('<b>Code package:</b> Aldawah_COSC4371_Checkpoint_Code.zip. Public repository URL pending publication; this local package does not yet satisfy the access-link requirement.','small')
p('1. Experimental design','heading')
p('<b>Question.</b> How much does detection change when classifiers trained on Safe-Guard are tested on Safe-Guard test, deepset and Gandalf, and how do obfuscation, normalization and the published ProtectAI reference affect that transfer?')
p('<b>(a) Variable.</b> The variable is the test dataset: Safe-Guard test, deepset or Gandalf, as assigned in the proposal feedback.')
p('<b>(b) What stays fixed.</b> Within each trained classifier, Safe-Guard training/validation rows, preprocessing, fitted weights and the validation-selected threshold stay fixed across test datasets; the test data changes. ProtectAI is a fixed pretrained reference, not a model trained exclusively on this project\'s split.')
p('<b>(c) Changes.</b> The approved question and models are unchanged. The baseline, full near-duplicate audit and split are now implemented. Recall at a validation false-positive-rate (FPR) target replaces the proposal\'s general metric list as the primary comparison; all three datasets are retained.')
p('<b>(d) Split design.</b> Dataset boundary is the main design: only Safe-Guard supplies training and validation. After the removals in 2(c), its released test partition stays test. Every released deepset and Gandalf partition is pooled into its own external test. Exact/near-text connected components are groups. The first fold of <font face="Courier">StratifiedGroupKFold(5, shuffle=True, random_state=4371)</font> assigns validation; the other four folds assign training (approximately 80/20). No future-time or unseen-user claim is made.')
p('<b>(e) Primary metric.</b> Recall at each classifier\'s threshold selected on the same Safe-Guard validation set for FPR &lt;=5%. This measures missed attacks and permits comparison across prevalence. The 5% target applies to validation, not as an external-test guarantee. Gandalf recall is interpreted as detection rate under a presumed-positive assumption.')
p('<b>(f) Main results table.</b> Each cell is clean recall; obfuscation drop (D); normalization recovery (N). Triplets are zero-width/homoglyph/base64. A dash marks work planned in Section 5, not a measured zero.','small')
rows=[['Classifier','Safe-Guard test','deepset','Gandalf*']]
for classifier,key in [('TF-IDF + logistic regression','lr'),('Fine-tuned DistilBERT','distilbert'),('ProtectAI v2 (reference)','protectai')]:
    cells=[]
    for ds in ['safeguard','deepset','gandalf']:
        value=f'{metric(ds):.4f}' if key=='lr' else '-'
        cells.append(f'Clean: {value}<br/>D: - / - / -<br/>N: - / - / -')
    rows.append([classifier]+cells)
table(rows,[145,127,127,127],9)
p('*Gandalf has no verified binary target. D = clean recall minus obfuscated recall; N = normalized-obfuscated recall minus obfuscated recall. All future variants keep the same source rows and threshold.','small')
p('2. Data and split','heading')
p('<b>(a) Loaded versions.</b> All Parquet files at pinned Hugging Face revisions: Safe-Guard [1] <font face="Courier">a3a877d608</font> (train/test); deepset [2] <font face="Courier">4f61ecb038</font> (train/test); Gandalf [3] <font face="Courier">04737b65e9</font> (train/validation/test). Full commit IDs, file paths, download URLs and SHA-256 hashes are in <font face="Courier">data_manifest.json</font>. Project 15 requires these three sources; the 2023-or-later-paper rule applies to Projects 9-12.','small')

page()
p('2. Data and split (continued)','heading')
p('<b>(b) Audit of loaded data.</b> Counts below were computed from all 11,958 rows before removal; Safe-Guard is the primary dataset. Exact duplicates mean identical original <font face="Courier">text</font>, excluding the label; excess means rows minus distinct text strings.','body')
table([
 ['Check','Safe-Guard','deepset','Gandalf'],
 ['Rows / source columns','10,296 / 2','662 / 2','1,000 / 2'],
 ['Negative class, n (%)','7,150 (69.44%)','399 (60.27%)','0 presumed (0%)'],
 ['Positive class, n (%)','3,146 (30.56%)','263 (39.73%)','1,000 presumed (100%)'],
 ['Missing source target','0','0','1,000: no binary label field'],
 ['Missing/blank text','0','0','0'],
 ['Exact duplicate excess / groups','159 / 91','0 / 0','0 / 0'],
 ['Near-duplicate pairs / involved rows','962 / 561','Not required','Not required'],
 ['Repeated entities / time range','Unavailable: no entity IDs or row timestamps','Same absence','Same absence'],
],[181,115,115,115],9)
p('Near-duplicate rule: Unicode NFKC, casefold and whitespace collapse; Jaccard similarity &gt;=0.85 between sets of character 5-grams (a shorter text is one token). Counts exclude identical raw-text pairs. A full, lossless prefix/length-filtered join, followed by exact intersection checks, ran on all datasets without sampling; connected components also group transitive chains. Unknown authors/sessions cannot be checked from text alone.','small')
p('Gandalf provides text and similarity, not binary ground truth; label=1 is assigned only for the checker and may include noisy positives. Similarity never enters the classifier. The source card describes July 2023 collection, but no per-row dates are available; no time split is used.','small')
p('<b>(c) Kept split counts.</b> Nonempty rows reproduce the checker. Zeros explicitly show that external data is never used for training/validation.','small')
rows=[['Dataset / split','Rows','Positive','Negative','Positive %']]
for ds in ['safeguard','deepset','gandalf']:
    for sp in ['train','validation','test']:
        x=next((x for x in r['split_counts'] if x['dataset']==ds and x['split']==sp),dict(rows=0,positive=0,negative=0))
        pct=f"{100*x['positive']/x['rows']:.2f}%" if x['rows'] else '-'
        rows.append([names[ds]+' / '+sp, f"{x['rows']:,}", f"{x['positive']:,}", f"{x['negative']:,}",pct])
table(rows,[206,80,80,80,80],8.7)
p('Removed 252 rows; 11,706 remain. In exclusive priority order: 15 rows with conflicting exact-text labels, 2 further cross-dataset exact-overlap rows, and 235 further Safe-Guard development rows in a component touching a test set. By source: 245 Safe-Guard train, 1 Safe-Guard test, 6 deepset, 0 Gandalf. No test row moved into development.','small')
p('Boundary counts overlap: 240 development rows touched a test component. Of these, 232 touched Safe-Guard test (79 direct exact matches, 153 near/transitive-only); 10 touched deepset (5 exact, 5 near/transitive-only); none touched Gandalf. Removing them protects development/test boundaries. The group split keeps remaining exact/near families together across training/validation; within-partition repeats stay. Other external exact duplicates were absent. Related author/session records beyond these text families are unavailable.','small')
table([['Aligned field','Safe-Guard','deepset','Gandalf'],['Model / duplicate field','text','text','text'],['Evaluation target','label','label','derived 1 (presumed); similarity excluded']],[140,112,112,162],8.7)
p('<b>(d) Leakage check.</b> Post-removal exact text matches: training vs. any test = <b>0</b> (0 for each dataset); training vs. validation = <b>0</b>. Exact/near-component overlap is also 0 for train/validation and for train or validation against every test set. These code assertions are in <font face="Courier">audit/results.json</font>; row-level removal reasons and group membership are retained.','small')

page()
p('3. Baseline and metrics','heading')
p(f'<b>(a) Pipeline.</b> Raw text -&gt; lowercase -&gt; default Unicode word tokenization (tokens of at least two word characters) -&gt; word unigram/bigram TF-IDF: min_df=2, max_features=100000, sublinear_tf=True, smooth_idf=True, L2 normalization -&gt; logistic regression: C=1, L2 penalty, liblinear, no class weights, tol=0.0001, max_iter=2000, seed=4371. Vocabulary/IDF/weights fit training only ({b["pipeline"]["fitted_vocabulary_size"]:,} features). Audit normalization is separate from model input. Fit took <b>{b["fit_seconds"]:.6f} seconds</b> on Windows CPU, Intel i7-13700K, one fit thread (16 physical cores/24 logical processors available). All-test scoring took {b["test_score_seconds"]:.6f} seconds.','small')
p(f'<b>(b) Threshold.</b> Positive=source-labeled injection (Gandalf: presumed injection). Positive iff score &gt;= <b>{threshold}</b>. Choose the lowest of the unique validation probabilities, plus one value above the maximum, with validation FPR &lt;=5%. Validation yields 55/1,117 false positives (4.9239%). The threshold is frozen before one pooled test-scoring call.','small')
p('<b>(c) Checker metrics.</b> Values and 95% intervals below match the supplied checker (1,000 row bootstrap resamples, seed 0). Test-set names match <font face="Courier">scores.csv</font>.','small')
rows=[['Test set','Test n','Positive n (%)','Random AP','Precision','Recall','FPR','AP','AUROC']]
for ds in ['safeguard','deepset','gandalf']:
    v=m[ds]
    rows.append([ds, str(v['test_rows']),f"{v['positive_rows']}<br/>({v['positive_percent']:.2f}%)",'one class' if ds=='gandalf' else f"{v['random_scorer_AP']:.4f}"]+[interval(ds,key) for key in ['precision','recall','false-positive rate','AP','AUROC']])
table(rows,[57,35,58,42,67,67,66,67,67],7.9)
p('For Gandalf, recall means presumed-positive detection rate; all other cells are one class. Row bootstrap intervals do not account for dependence among repeated templates, so uncertainty may be understated.','small')
p(f'<b>(d) Random reference.</b> Safe-Guard AP {metric("safeguard","AP"):.4f} exceeds {m["safeguard"]["random_scorer_AP"]:.4f} and AUROC {metric("safeguard","AUROC"):.4f} exceeds 0.5; deepset AP {metric("deepset","AP"):.4f} exceeds {m["deepset"]["random_scorer_AP"]:.4f} and AUROC {metric("deepset","AUROC"):.4f} exceeds 0.5. Both beat both random references. Gandalf has one class.','small')
p('<b>(e) Too-good check.</b> Safe-Guard recall, AP and AUROC exceed 0.99; the executed original-text and exact/near-family leakage assertions found zero training/validation/test overlaps, and only text entered the classifier. This resolves the checker\'s requested check, not every possible shortcut: synthetic templates and collection style could still make Safe-Guard unusually easy.','small')
p('<b>(f) Reproducibility.</b> Python 3.12.14; package versions are pinned in requirements.txt (scikit-learn 1.7.2). Install it, then run <font face="Courier">python run_checkpoint.py</font>. The command downloads SHA-256-verified pinned data, audits, trains, writes split.csv/scores.csv and runs the unchanged checker. Seed=4371. Raw data is downloaded locally, not redistributed. '+('Repository: '+link(url,url)+'.' if url else 'Repository URL is pending publication; the complete local code package is attached.'),'small')
p(f'<font face="Courier">python check_checkpoint.py --split split.csv --scores scores.csv<br/>--threshold &quot;{threshold}&quot;</font> (one command line)','small')
p('4. Figure','heading')
blocks.append({'kind':'image','path':'repository/figures/deepset_confusion_matrix.png','width':258,'height':180.6})
p('Figure 1. Baseline confusion matrix on deepset: 656 test rows; 258 positive (39.33%). Cells show counts; threshold='+threshold+'.','caption')
p('At the validation-calibrated threshold, the model catches 106/258 deepset injections (41.09%), showing weak transfer despite its high Safe-Guard recall; deepset FPR is 24/398 (6.03%).','small')

page()
p('5. Plan to finish','heading')
p('<b>(a) Required elements and target completion dates.</b> Dates below are a proposed remaining-work schedule, not a claim that later experiments have run.','small')
table([
 ['Required element','Status','Finish date'],
 ['Three datasets, full audit, group-safe split, LR baseline and checker outputs','done','October 29, 2026'],
 ['Review 30 records per available class/source, remaining label noise and text shortcuts','not started','October 31, 2026'],
 ['Fine-tune DistilBERT on the same Safe-Guard training rows; calibrate on the same validation rows','not started','November 2, 2026'],
 ['Run ProtectAI v2 reference; calibrate on Safe-Guard validation; record training-exposure limits','not started','November 3, 2026'],
 ['Three fixed obfuscations: zero-width characters, homoglyph substitutions, base64; score each model and report each drop','not started','November 4, 2026'],
 ['Validation-frozen blind normalization; recovery for each obfuscation; benign FPR and truncation checks','not started','November 5, 2026'],
 ['Complete classifier-by-dataset table, paired variant comparisons, limitations and final reproducibility pass','started','November 6, 2026'],
],[350,72,104],9)
p('<b>(b) Main comparison.</b> Compare test-dataset values Safe-Guard test, deepset and Gandalf for each classifier. The Safe-Guard training/validation rows are unchanged, so no alternate validation set is needed. Each model uses its own threshold chosen on that same validation set at FPR &lt;=5%; the model and threshold stay fixed across tests. ProtectAI\'s pretrained weights remain fixed, with prior exposure unresolved. Primary metric=recall (Gandalf: presumed-positive detection rate).')
H=b['primary_metric_widest_ci_half_width']; delta=0.08
p(f'Let D = Safe-Guard recall minus external-dataset recall. The widest baseline recall interval is deepset [0.3512, 0.4706]; using unrounded limits, its halfwidth is <b>{H:.8f}</b> (5.9688 percentage points), larger than either other test interval. Freeze a material-loss cutoff of <b>0.08 (8 percentage points)</b>, which exceeds that halfwidth. A <b>yes</b> is D &gt;=0.08; a <b>no detected material loss</b> is D &lt;0.08, evaluated separately for each external set. This is an effect-size screen, not a significance test or proof of equivalence.')
dd=metric('safeguard')-metric('deepset'); dg=metric('safeguard')-metric('gandalf')
p(f'The baseline gives D={dd:.4f} for deepset (yes) and D={dg:.4f} for Gandalf (no under this rule). The Gandalf result remains conditional on its noisy positive labels. For each classifier and dataset, separately report the three obfuscation drops and three normalization recoveries defined in 1(f), holding original row IDs, model and threshold fixed. Select the normalization policy using validation variants only, apply it without the true class label, and check benign FPR. Do not tune on external-test outcomes.','small')
B=b['fit_seconds']; seconds=(20000+14*750+6+1+3000)*B
p('<b>(c) Remaining compute budget.</b> The measured baseline fit is B='+f'{B:.7f}'+' seconds. Budget one DistilBERT fit at 20,000B; 14 transformer scoring batches (two models x clean plus six altered/normalized views, each covering all three test sets) at 750B each; six additional LR scoring batches at B each; one reproduction fit at B; and 3,000B for the transformer pilot, validation scoring, threshold calibration and validation-variant normalization selection. Total = (20,000 + 14 x 750 + 6 + 1 + 3,000)B = '+f'<b>{seconds/60:.2f} minutes</b> (about {round(seconds/60)} minutes).','small')
p('These are provisional CPU budgeting multipliers, not measured transformer runtimes: LR timing alone cannot predict them. Benchmark a short transformer pilot before the full run and revise the budget. Downloads, manual review and report editing are outside this run-time estimate.','small')
p('Sources and assistance','heading')
refs=[
 ('[1] xTRam1. Safe-Guard prompt-injection dataset; pinned revision above.','https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection'),
 ('[2] deepset. Prompt-injections dataset; pinned revision above.','https://huggingface.co/datasets/deepset/prompt-injections'),
 ('[3] Lakera. Gandalf ignore-instructions dataset; presumed positives selected automatically.','https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions'),
 ('[4] Protect AI. DeBERTa-v3-base-prompt-injection-v2 model card.','https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2'),
]
for text,u in refs: p(text+' '+link(u,'Source link')+'.','small')
p('Data limits: Safe-Guard\'s card does not declare a license; deepset\'s license metadata is inconsistent. This package distributes code, hashes, split assignments and scores, with raw-data download instructions. ProtectAI and Safe-Guard share some named source ancestry; exact benchmark exposure is unknown.','small')
p('[5] OpenAI. ChatGPT/Codex, October 29, 2026. '+link('https://chatgpt.com/','https://chatgpt.com/')+'. AI assistance was used for formatting and verification purposes.','small')
(ROOT/'report_content.json').write_text(json.dumps(blocks,indent=2),encoding='utf-8')
print('Prepared',len(blocks),'blocks; estimated remaining compute',seconds/60,'minutes')
