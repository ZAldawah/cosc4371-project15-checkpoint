"""Build the final report directly from checker-derived results."""
from pathlib import Path
import html
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted, KeepTogether

HERE=Path(__file__).resolve().parent
REPO='https://github.com/ZAldawah/cosc4371-project15-checkpoint'
NAMES={'tfidf_lr':'TF-IDF + LR','distilbert':'DistilBERT','protectai':'ProtectAI v2',
       'safeguard':'Safe-Guard','deepset':'deepset','gandalf':'Gandalf*'}
MODELS=['tfidf_lr','distilbert','protectai']

def read(path): return json.loads((HERE/path).read_text(encoding='utf-8'))
def f(value): return 'n/a' if value is None else f'{value:.4f}'
def pct(value): return f'{100*value:.2f}%'

def main():
    main_rows=read('results/main_metrics.json'); extra=read('results/obfuscation_metrics.json')
    zero=read('results/zeroshot_metrics.json'); errors=read('results/error_summary.json')
    rules=read('audit/error_category_rules.json')['rules_in_priority_order']
    lr=read('results/tfidf_lr_metadata.json'); db=read('audit/transformer_distilbert.json')
    pa=read('audit/transformer_protectai.json'); zs=read('audit/zeroshot_results.json')
    verification=read('audit/final_data_verification.json'); labels=read('audit/label_review_summary.json')
    assert len(main_rows)==9 and db['status']==pa['status']=='complete'
    cells={(r['value'],r['column']):r for r in main_rows}
    supplemental={(r['value'],r['column']):r for r in extra}
    styles={
        'title':ParagraphStyle('title',fontName='Times-Bold',fontSize=16,leading=18,spaceAfter=9,textColor=colors.black),
        'h':ParagraphStyle('h',fontName='Times-Bold',fontSize=12,leading=14,spaceBefore=8,spaceAfter=5,textColor=colors.black),
        'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=12.4,spaceAfter=6,textColor=colors.black),
        'small':ParagraphStyle('small',fontName='Times-Roman',fontSize=8.6,leading=10.5,spaceAfter=5,textColor=colors.black),
        'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=8.5,leading=10.1,textColor=colors.black),
        'tiny':ParagraphStyle('tiny',fontName='Times-Roman',fontSize=7.7,leading=9.2,spaceAfter=4,textColor=colors.black),
        'mono':ParagraphStyle('mono',fontName='Courier',fontSize=7.2,leading=9.0,textColor=colors.black)
    }
    story=[]
    def p(text,kind='body'): story.append(Paragraph(text,styles[kind]))
    def h(text): p(text,'h')
    def table(rows,widths,font='cell'):
        converted=[[Paragraph(str(x),styles[font]) for x in row] for row in rows]
        t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
              ('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,-1),(-1,-1),.4,colors.black),
              ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
              ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,6)])
    def metric_cell(row):
        m=row['metrics']['recall']; lo,hi=m['ci95']
        return f"{m['value']:.4f}<br/>[{lo:.4f}, {hi:.4f}]"

    # Figure required by Section 2: one classifier column, all three dataset values.
    fig,ax=plt.subplots(figsize=(6.8,2.25),layout='constrained')
    selected=[cells[(v,'tfidf_lr')] for v in ['safeguard','deepset','gandalf']]
    values=np.array([r['metrics']['recall']['value'] for r in selected])
    intervals=np.array([r['metrics']['recall']['ci95'] for r in selected])
    ax.errorbar(values,[2,1,0],xerr=np.array([values-intervals[:,0],intervals[:,1]-values]),fmt='o',color='#205c91',capsize=5)
    for i,r in enumerate(selected): ax.annotate(f"{r['metrics']['recall']['value']:.4f}",(values[i],2-i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.set_yticks([2,1,0],['Safe-Guard (reference)','deepset','Gandalf*'])
    ax.set_xlim(0,1.05); ax.set_ylim(-.45,2.55); ax.set_xlabel('Recall at the fixed TF-IDF + LR threshold')
    ax.grid(axis='x',alpha=.22); ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'figures'/'main_recall_intervals.png',dpi=220); plt.close(fig)

    p('Final Checkpoint: Prompt-Injection Detection Across Datasets','title')
    p('Zainulabdeen Aldawah | COSC 4371 | Project 15 | November 20, 2026','small')
    p(f'Repository: <link href="{REPO}">{REPO}</link>','small')
    h('1. Main comparison')
    p('<b>Question.</b> How much does recall decrease across Safe-Guard, deepset and Gandalf for Safe-Guard-trained classifiers and a calibrated pretrained reference, and how do fixed obfuscations and normalization affect detection?')
    p('<b>(a) What ran.</b> The variable remains the <b>test dataset</b>: Safe-Guard test (reference), deepset, and Gandalf. Columns are TF-IDF/logistic regression, fine-tuned DistilBERT, and the published ProtectAI v2 reference. The final handout reverses the midsemester table orientation; the underlying comparison is unchanged. Every model and its threshold stay fixed across test datasets. All original clean cells and all three obfuscation/recovery comparisons are now measured.')
    table([['Fixed partition','Rows','Positive / negative'],['Safe-Guard training','6,400','1,917 / 4,483'],['Safe-Guard validation','1,591','474 / 1,117'],['Safe-Guard test','2,059','650 / 1,409'],['deepset test','656','258 / 398'],['Gandalf test*','1,000','1,000 presumed / 0']],[250,70,200])
    p('All released deepset and Gandalf partitions are external tests. A publisher split name such as "train" inside a row ID does not make it a project training row.','tiny')
    p('No primary test rows were subsampled. Seed 4371 controls the group split, model training and error sample. Changes since the midsemester are the completed transformer runs, measured timing pilot, blind normalization controls, a supplementary Gandalf-plus-benign pool, and a Week 10 zero-shot comparison. They address the instructor feedback without replacing the dataset-transfer question.','small')
    p('<b>(b) Thresholds.</b> Each classifier uses the lowest unique clean Safe-Guard validation score (plus a candidate above the maximum) with validation FPR at most 5%. Positive means score at or above the threshold. The same threshold is used for every test dataset, obfuscation and defense. The 5% target is not an external-test guarantee.','small')
    table([['Classifier','Exact threshold','Validation FP / benign'],
           [NAMES['tfidf_lr'],lr['threshold_repr'],'55 / 1,117'],
           [NAMES['distilbert'],db['threshold_repr'],f"{db['validation']['false_positives']} / 1,117"],
           [NAMES['protectai'],pa['threshold_repr'],f"{pa['validation']['false_positives']} / 1,117"]],[130,250,140])
    p('<b>(c) Leakage.</b> LR and DistilBERT use the <b>same training set</b>. A fresh full character-5-gram Jaccard audit (NFKC/casefold/whitespace normalization, similarity at least 0.85) reproduced the midsemester split. Exact training-text matches against validation and each test set: <b>0</b>; training/validation exact-or-near family overlaps and development/test family overlaps: <b>0</b>. These are measured text checks, not a guarantee against all shared templates or pretraining exposure.','small')
    p('ProtectAI has externally trained weights: its two corresponding training-overlap counts are <b>unavailable</b> because the full training corpus is not provided. Local calibration used only the project validation rows. Metrics reaching 0.99 are discussed with these checks; high ProtectAI results cannot be treated as verified unseen-data performance. Evidence: audit/final_data_verification.json and audit/transformer_protectai.json.','small')

    story.append(PageBreak())
    h('1. Main comparison (continued)')
    p('<b>(d) Results.</b> Recall and 95% row-bootstrap intervals below come from the unchanged final checker (1,000 resamples per cell). All nine primary cells are complete.','small')
    table([['Test-dataset value']+[NAMES[m] for m in MODELS]]+
          [[NAMES[v]]+[metric_cell(cells[(v,m)]) for m in MODELS] for v in ['safeguard','deepset','gandalf']],[130,130,130,130])
    p('<b>False alarms on deepset:</b> '+', '.join(f"{NAMES[m]} {pct(cells[('deepset',m)]['metrics']['false-positive rate']['value'])} ({cells[('deepset',m)]['fp']}/398)" for m in MODELS)+'. DistilBERT\'s higher recall than LR comes with substantially more false alarms; recall alone does not establish better practical detection.','small')
    random_cells=[f"{NAMES[r['value']]} / {NAMES[r['column']]}" for r in main_rows if r['negative'] and (r['metrics']['AP']['value']<=r['positive']/r['rows'] or r['metrics']['AUROC']['value']<=.5)]
    p('Cells with AP at or below positive prevalence or AUROC at or below 0.5: '+(', '.join(random_cells)+'. Dataset/label-domain mismatch may reverse score ordering.' if random_cells else '<b>none</b> among the two-class primary cells.')+' Gandalf has only presumed positives; only detection rate (checker recall) is interpretable.','small')
    p('<b>(e) Verdict.</b> Direction: <b>lower recall</b>. Change D = Safe-Guard recall minus external-dataset recall within the same column. The midsemester yes cutoff remains D at least 0.08. Its strict no rule (D &lt; 0.08) is expressed as no at most 0.0799, with an explicit 0.0001 undecided gap to follow the final handout\'s closed inequalities. This boundary clarification was fixed before the new transformer results; decisions use unrounded values. These are practical-importance rules, not significance or equivalence tests.','small')
    table([['Column','External value','Checker change D','Verdict']]+
          [[NAMES[m],NAMES[v],f"{cells[(v,m)]['change']:+.4f}",cells[(v,m)]['verdict']] for m in MODELS for v in ['deepset','gandalf']],[150,130,130,110])
    h('2. Figure')
    story.append(Image(str(HERE/'figures'/'main_recall_intervals.png'),width=500,height=165.44))
    p('Figure 1. TF-IDF + LR column: Safe-Guard reference, 2,059 test rows, 31.57% positive; deepset, 656 rows, 39.33% positive; Gandalf, 1,000 rows, 100% presumed positive. Points show recall and bars show the checker\'s 95% intervals. All retained original test rows are included.','small')
    r=cells[('deepset','tfidf_lr')]
    p(f"The Safe-Guard-to-deepset recall loss is {r['change']:.4f} ({100*r['change']:.2f} percentage points), exceeding the prespecified 8-point material-loss cutoff.",'small')

    story.append(PageBreak())
    h('3. Claim and limitation')
    yes=[f"{NAMES[r['value']]} / {NAMES[r['column']]} (D={r['change']:.4f})" for r in main_rows if r['verdict']=='yes']
    no=[f"{NAMES[r['value']]} / {NAMES[r['column']]}" for r in main_rows if r['verdict']=='no']
    claim='Material recall loss relative to the same-column Safe-Guard reference is supported for '+', '.join(yes)+'.' if yes else 'The measured external cells show no material recall loss under the prespecified decision rule.'
    if all(r['verdict']=='undecided' for r in main_rows if r['value']!='safeguard'): claim='Inconclusive under the prespecified decision rule.'
    p('<b>(a) Claim.</b> '+claim)
    p('<b>(b) Limitation and settling check.</b> Source differences combine language, prompt style and uncertain task labels: Gandalf positives are presumed, and deepset includes multilingual and context-dependent prompts. These factors could change the apparent transfer loss. A check that would settle whether the claim persists is an independently labeled, application-context-preserving sample from each source, stratified to matched language and prompt-length ranges, with thresholds frozen and uncertainty resampled by text family. The current row intervals may be optimistic for repeated templates.')
    p('The 150-row qualitative review (30 per available source/class, seed 4371) found 104 descriptions consistent with their source labels, 16 context-ambiguous cases, and 30 Gandalf cases that remain presumed-only. It exposed structured Safe-Guard benign templates, explicit override language, and multilingual deepset prompts. These are nonblind descriptive judgments, not estimates of population label accuracy; no source label or model setting changed. See audit/label_review.csv.','small')
    h('4. Error analysis')
    p(f"<b>(a) Sample.</b> Cell: <b>deepset / TF-IDF + LR</b>, clean text. Of {errors['population']['FP']} false positives and {errors['population']['FN']} false negatives, 10 of each were drawn without replacement after sorting row IDs, using NumPy default_rng(4371), false positives first. The cell contains both classes. errors.csv records all 20 selected IDs.")
    cat={r['category']:r for r in errors['categories']}
    table([['Category and one-sentence rule','FP','FN','Example row ID']]+
          [[f"<b>{rule['name']}.</b> {html.escape(rule['rule'])}",cat[rule['category']].get('FP',0),cat[rule['category']].get('FN',0),cat[rule['category']]['example_row_id']] for rule in rules],[310,30,30,150])
    p('<b>(b) Interpretation.</b> Categories are applied in the displayed priority order, so every sampled row has exactly one category; there is no catch-all. Non-English or mixed-language text accounts for 4 sampled false positives and 7 sampled false negatives. English factual/political questions account for the other false alarms. The remaining misses use task/output redirection or letter spacing. These observations suggest vocabulary and context mismatch; they do not prove the cause of individual predictions.','small')

    story.append(PageBreak())
    h('5. Reproducibility')
    p(f'<b>(a) Files and command.</b> Repository: <link href="{REPO}">{REPO}</link>. The final package includes split.csv, scores.csv and errors.csv; pinned input/model manifests; all scripts; validation probabilities; audit evidence; raw per-model probabilities; and figures. The orchestration command below writes the scored rows and rebuilds this report. Public artifacts contain IDs, labels, hashes and scores; raw datasets and pretrained weights are downloaded separately, and fine-tuned weights are generated locally.')
    p('<font face="Courier">python run_final.py</font><br/>Uses the data/ cache, downloading pinned Parquet files when absent. Install requirements.txt in Python 3.12 first; the README gives CUDA and CPU setup. To reproduce only the final checker:','small')
    p('<font face="Courier">python check_final_checkpoint.py --split split.csv --scores scores.csv --errors errors.csv --metric recall --reference safeguard --direction lower</font>','small')
    p('LR uses the unchanged word unigram/bigram TF-IDF pipeline (min_df=2, max_features=100000, sublinear TF) and logistic regression (C=1, liblinear, seed 4371). DistilBERT uses all 6,400 training rows for 3 fixed epochs, learning rate 2e-5, AdamW weight decay 0.01, effective batch 16, 10% warmup, and maximum 512 tokens; the final epoch is fixed in advance. ProtectAI weights stay frozen. Model revisions, parameters and package versions are recorded in the audit files.','small')
    p('A fresh data audit reproduced the frozen split; the new LR probabilities matched every midsemester clean-test probability exactly. GPU training is seeded but emitted a nondeterministic attention-backward warning, so bit-for-bit retraining is not claimed. Saved probabilities and checker results are deterministic artifacts. Exact-input inference caching is label blind and preserves one score row per original observation and view.','small')
    p('<b>(b) Checker output.</b> Appendix B contains the complete primary-checker output, including warnings. Section 1 values and changes use that checker\'s functions and four-decimal formatting. Structural exit code 0 does not establish scientific validity; the ceiling warnings are addressed in 1(c), and the source/label limits remain.','small')
    h('6. Plan to finish')
    p('<b>(a) Required elements.</b> The midsemester required-elements table is updated below. All its elements and all originally planned classifier-by-dataset cells are complete; there are no unfinished cells requiring future finish dates.','small')
    table([['Midsemester required element','Status / completion date'],
      ['Three datasets, full audit, group-safe split, LR baseline and checker outputs','Done / Nov. 20, 2026'],
      ['Review 30 records per available class/source; label noise and text shortcuts','Done / Nov. 20, 2026'],
      ['Fine-tune DistilBERT on the same training rows; calibrate on validation','Done / Nov. 20, 2026'],
      ['Run ProtectAI v2; calibrate on validation; record exposure limits','Done / Nov. 20, 2026'],
      ['Zero-width, homoglyph and Base64 tests; each classifier and each drop','Done / Nov. 20, 2026'],
      ['Validation-frozen blind normalization, recovery, benign FPR and truncation','Done / Nov. 20, 2026'],
      ['Complete model/dataset table, paired variants, limitations, reproducibility','Done / Nov. 20, 2026']],[385,135])
    training_minutes=db['training_seconds']/60
    p(f"<b>(b) Compute.</b> Remaining required model runs: <b>0 minutes</b>. LR fit measured {lr['fit_seconds']:.3f} seconds. The first 20 DistilBERT microbatches (160 examples) took {db['pilot']['seconds']:.2f} seconds; 2,400 total microbatches projected {db['pilot']['projected_full_training_seconds']/60:.2f} minutes. The actual three-epoch fit took <b>{training_minutes:.2f} minutes</b> on an RTX 3060 Ti. All-view scoring took {db['test_scoring_seconds']/60:.2f} minutes for DistilBERT and {pa['test_scoring_seconds']/60:.2f} minutes for ProtectAI, with exact-text caching. Zero-shot inference took {zs['inference_seconds_including_class_descriptions']/60:.2f} minutes. Downloads, report editing and the separate data audit are outside these training/inference times.",'small')

    story.append(PageBreak())
    h('Appendix A. Supporting experiments and sources')
    p('<b>Obfuscation and recovery on deepset.</b> All 656 original rows (258 positive, 398 benign) receive each transformation without using their labels. Zero-width inserts U+200B between adjacent ASCII letters/digits; homoglyph substitutes 15 fixed Latin lookalikes; Base64 wraps the entire UTF-8 text with a fixed decode instruction. These are controlled lexical stress tests, not verified successful downstream attacks.','small')
    table([['Model','Transform','Attacked<br/>recall / FPR','Normalized<br/>recall / FPR','Recall<br/>drop','Recall<br/>recovery']]+
      [[NAMES[m],{'zero_width':'Zero-width','homoglyph':'Homoglyph','base64':'Base64'}[v],
        f(supplemental[(v,'deepset__'+m)]['metrics']['recall']['value'])+' / '+f(supplemental[(v,'deepset__'+m)]['metrics']['false-positive rate']['value']),
        f(supplemental[('normalized_'+v,'deepset__'+m)]['metrics']['recall']['value'])+' / '+f(supplemental[('normalized_'+v,'deepset__'+m)]['metrics']['false-positive rate']['value']),
        f(supplemental[('clean','deepset__'+m)]['metrics']['recall']['value']-supplemental[(v,'deepset__'+m)]['metrics']['recall']['value']),
        f(supplemental[('normalized_'+v,'deepset__'+m)]['metrics']['recall']['value']-supplemental[(v,'deepset__'+m)]['metrics']['recall']['value'])] for m in MODELS for v in ['zero_width','homoglyph','base64']],[100,75,105,105,65,70],font='tiny')
    p('<b>Reading the signs.</b> Several transformer obfuscation cells flag nearly every benign prompt (FPR 0.9673-1.0000), while recall reaches 1. A negative recall drop is therefore not an improvement in practical detection. On deepset, normalization restores the clean operating point: its negative recall recovery accompanies a large reduction in false alarms. On Safe-Guard, DistilBERT normalization changes one additional benign prediction to positive (82 to 83 false alarms out of 1,409).','tiny')
    p('The blind normalizer uses NFKC, removes four fixed invisible characters, reverses the known lookalikes and strictly decodes the known Base64 wrapper. It was accepted using validation only, then frozen. Normalized-clean is a separate control. Designed reversibility explains recovery under these encodings; no claim of general attack robustness follows. All classifier/dataset drops, recoveries and FPRs are in results/obfuscation_changes.csv; complete intervals are in results/obfuscation_metrics.json and truncation counts in the transformer audits.','tiny')
    mixed=[supplemental[('clean','gandalf_mixed__'+m)] for m in MODELS]
    p('<b>Gandalf plus benign control.</b> All 1,000 presumed Gandalf positives are paired with all 1,409 held-out Safe-Guard benign rows (2,409 total; 41.51% positive). This is an artificial cross-source pool, not demonstrated style matching or original Gandalf negatives. Clean FPR is '+', '.join(f"{NAMES[m]} {pct(r['metrics']['false-positive rate']['value'])}" for m,r in zip(MODELS,mixed))+'. All eight views retain both classes. Reused benign rows make this control dependent on Safe-Guard results.','tiny')
    zero_by={r['value']:r for r in zero}
    p('<b>Week 10 zero-shot check.</b> Following Lecture 1, pages 18 and 31-36, use L2-normalized embeddings and nearest class-description cosine similarity. Qwen3-Embedding-0.6B replaces the lecture\'s 8B model to fit local resources; fixed descriptions distinguish a normal request from instruction override/extraction. The score is cosine(injection) minus cosine(benign), threshold 0, maximum 1,024 tokens, with no labeled training or calibration.','tiny')
    p('Zero-shot recall: '+', '.join(f"{NAMES[v]} {f(zero_by[v]['metrics']['recall']['value'])}" for v in ['safeguard','deepset','gandalf'])+f"; deepset FPR {f(zero_by['deepset']['metrics']['false-positive rate']['value'])}. This comparison checks transfer without Safe-Guard fine-tuning; it cannot identify a causal fraction of the synthetic-data ceiling. Exact descriptions, intervals and model revision are recorded in the zero-shot audit/results.",'tiny')
    sources=[
      ('Safe-Guard dataset','https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection'),
      ('deepset prompt-injections','https://huggingface.co/datasets/deepset/prompt-injections'),
      ('Lakera Gandalf','https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions'),
      ('DistilBERT base','https://huggingface.co/distilbert/distilbert-base-uncased'),
      ('ProtectAI v2 model card','https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2'),
      ('Qwen3-Embedding-0.6B','https://huggingface.co/Qwen/Qwen3-Embedding-0.6B')]
    p('<b>Sources.</b> '+ '; '.join(f'[{i}] <link href="{url}">{title}</link>' for i,(title,url) in enumerate(sources,1))+'. [7] R. Verma and B. Tuck, COSC 4371 Week 10 L1 slides and demo, October 27, 2026. [8] Final Project Checkpoint, Fall 2026. Dataset hashes and full revisions are in data_manifest.json; model revisions are in their audits.','tiny')
    p('<b>AI disclosure.</b> AI was used in the organization, formatting, and structuring of this document.','small')

    story.append(PageBreak())
    h('Appendix B. Full primary-checker output')
    output=(HERE/'checker_output.txt').read_text(encoding='utf-8')
    for block in output.strip().split('\n\n'):
        lines=[]
        for line in block.splitlines():
            lines.extend(textwrap.wrap(line,width=115,replace_whitespace=False,drop_whitespace=False) or [''])
        story.append(KeepTogether([Preformatted('\n'.join(lines),styles['mono']),Spacer(1,8)]))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFillColor(colors.black);canvas.setFont('Times-Roman',8)
        canvas.drawString(46,24,'COSC 4371 | Project 15 | Zainulabdeen Aldawah')
        canvas.drawRightString(566,24,str(doc.page));canvas.restoreState()
    target=HERE/'Aldawah_COSC4371_Final_Checkpoint.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=letter,rightMargin=46,leftMargin=46,topMargin=38,bottomMargin=38,
                          title='COSC 4371 Project 15 Final Checkpoint',author='Zainulabdeen Aldawah')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target,flush=True)

if __name__=='__main__': main()
