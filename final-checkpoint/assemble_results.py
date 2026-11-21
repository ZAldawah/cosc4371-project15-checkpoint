"""Assemble final CSV contracts and checker-derived summaries from measured model outputs."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import check_final_checkpoint as checker
from final_common import HERE, SEED, VIEWS, write_json

MODELS = ['tfidf_lr','distilbert','protectai']

def sampled_errors(scores):
    cell = scores[(scores.value=='deepset') & (scores.column=='tfidf_lr')].copy()
    cell['error_type'] = np.where(cell.label==0,'FP','FN')
    wrong = cell[(cell.score>=cell.threshold) != cell.label.astype(bool)]
    rng = np.random.default_rng(SEED)
    sampled = []
    for kind in ['FP','FN']:
        pool = wrong[wrong.error_type==kind].sort_values('row_id').reset_index(drop=True)
        sampled.append(pool.iloc[rng.choice(len(pool),min(10,len(pool)),replace=False)])
    sampled = pd.concat(sampled,ignore_index=True)
    annotations = pd.read_csv(HERE/'audit'/'error_annotations.csv')
    assert set(annotations.row_id)==set(sampled.row_id)
    sampled = sampled.merge(annotations,on='row_id',validate='one_to_one',sort=False)
    sampled[['row_id','value','column','category']].to_csv(HERE/'errors.csv',index=False)
    counts = sampled.groupby(['category','error_type']).size().unstack(fill_value=0).reset_index()
    counts['example_row_id'] = counts.category.map(sampled.groupby('category').row_id.first())
    write_json(HERE/'results'/'error_summary.json',{'seed':SEED,'population':{'FP':int((wrong.error_type=='FP').sum()),'FN':int((wrong.error_type=='FN').sum())},'sampled':{'FP':int((sampled.error_type=='FP').sum()),'FN':int((sampled.error_type=='FN').sum())},'categories':counts.to_dict('records')})

def run_checker(path,output,reference,errors=False):
    command=[sys.executable,str(HERE/'check_final_checkpoint.py'),'--split',str(HERE/'split.csv'),
             '--scores',str(path),'--metric','recall','--reference',reference,'--direction','lower']
    if errors:
        command += ['--errors',str(HERE/'errors.csv')]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    output.write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result

def summarize(scores, ci=True):
    out=[]
    for (value,column),part in scores.groupby(['value','column'],sort=False):
        part=part.sort_values('row_id',kind='mergesort')
        y=part.label.to_numpy(); s=part.score.to_numpy(); threshold=float(part.threshold.iloc[0]); pred=s>=threshold
        values=checker.all_metrics(y,s,pred)
        intervals=checker.bootstrap(y,s,pred,checker.group_rng(value,column),checker.reps_for(len(y))) if ci else {}
        metrics={k:{'value':None if v is None else float(v),'ci95':intervals.get(k)} for k,v in values.items()}
        out.append({'value':value,'column':column,'rows':len(part),'positive':int(y.sum()),'negative':int((y==0).sum()),
                    'threshold':threshold,'tp':int(((y==1)&pred).sum()),'fp':int(((y==0)&pred).sum()),
                    'tn':int(((y==0)&~pred).sum()),'fn':int(((y==1)&~pred).sum()),'metrics':metrics})
    return out

def assemble_zero_shot():
    zero=pd.read_csv(HERE/'results'/'zeroshot_scores.csv').rename(columns={'dataset':'value'})
    zero['column']='qwen_embedding_zeroshot'
    zero=zero[['row_id','value','column','label','score','threshold']]
    zero.to_csv(HERE/'scores_zeroshot.csv',index=False,float_format='%.17g')
    zero=pd.read_csv(HERE/'scores_zeroshot.csv')
    run_checker(HERE/'scores_zeroshot.csv',HERE/'checker_zeroshot_output.txt','safeguard')
    write_json(HERE/'results'/'zeroshot_metrics.json',summarize(zero))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--skip-checkers',action='store_true')
    parser.add_argument('--zero-shot-only',action='store_true',help='Assemble a newly completed supplemental zero-shot run without repeating other bootstraps')
    args=parser.parse_args()
    if args.zero_shot_only:
        assemble_zero_shot()
        return
    all_scores=[]
    for model in MODELS:
        path=HERE/'results'/f'{model}_scores.csv'
        if not path.exists():
            if args.allow_partial: continue
            raise FileNotFoundError(path)
        rows=pd.read_csv(path,float_precision='round_trip')
        if model!='tfidf_lr':
            audit=json.loads((HERE/'audit'/f'transformer_{model}.json').read_text(encoding='utf-8'))
            assert audit['status']=='complete', f'{model} run is incomplete'
        rows['model']=model
        assert set(rows['view'])==set(VIEWS), model
        all_scores.append(rows)
    raw=pd.concat(all_scores,ignore_index=True)
    split=pd.read_csv(HERE/'split.csv')
    original=split[split.split=='test'].set_index('row_id')
    for (model,view),part in raw.groupby(['model','view']):
        assert part.row_id.is_unique
        assert set(part.row_id)==set(original.index)
        assert np.array_equal(part.label,original.loc[part.row_id,'label'])
        assert part.threshold.nunique()==1
    for model,part in raw.groupby('model'):
        assert part.threshold.nunique()==1, 'Test views must share a model threshold'
    clean=raw[raw['view']=='clean'].rename(columns={'dataset':'value','model':'column'})
    clean=clean[['row_id','value','column','label','score','threshold']]
    clean.to_csv(HERE/'scores.csv',index=False,float_format='%.17g')
    # Use the supplied checker's own CSV parser for metrics, including threshold roundoff.
    clean=pd.read_csv(HERE/'scores.csv')
    sampled_errors(clean)
    supplemental=[]
    for model,part in raw.groupby('model',sort=False):
        for dataset,group in part.groupby('dataset',sort=False):
            item=group.copy(); item['value']=item['view']; item['column']=f'{dataset}__{model}'
            supplemental.append(item)
        mixed=part[(part.dataset=='gandalf')|((part.dataset=='safeguard')&(part.label==0))].copy()
        mixed['value']=mixed['view']; mixed['column']=f'gandalf_mixed__{model}'
        supplemental.append(mixed)
    supplemental=pd.concat(supplemental,ignore_index=True)[['row_id','value','column','label','score','threshold']]
    supplemental.to_csv(HERE/'scores_obfuscation.csv',index=False,float_format='%.17g')
    supplemental=pd.read_csv(HERE/'scores_obfuscation.csv')
    if not args.skip_checkers:
        run_checker(HERE/'scores.csv',HERE/'checker_output.txt','safeguard',errors=True)
    main_metrics=summarize(clean)
    design=json.loads((HERE/'audit'/'design.json').read_text())
    for row in main_metrics:
        reference=next(r for r in main_metrics if r['value']=='safeguard' and r['column']==row['column'])
        change=reference['metrics']['recall']['value']-row['metrics']['recall']['value']
        row['change']=change
        row['verdict']='reference' if row['value']=='safeguard' else 'yes' if change>=design['yes_difference'] else 'no' if change<=design['no_difference'] else 'undecided'
    write_json(HERE/'results'/'main_metrics.json',main_metrics)
    # Supplement intervals use the exact same checker functions and seeds; main full output is in the PDF.
    supplemental_metrics=summarize(supplemental,ci=False)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        checker.warnings.clear()
        res=checker.classifier_metrics(supplemental)
        checker.results_table(res,list(supplemental.value.unique()),list(supplemental.column.unique()),'recall','clean','lower')
        checker.section('Warnings')
        for warning in checker.warnings:
            print('- '+warning)
        if not checker.warnings: print('None.')
    for row in supplemental_metrics:
        for metric,interval in res[(row['value'],row['column'])]['ci'].items():
            row['metrics'][metric]['ci95']=interval
    write_json(HERE/'results'/'obfuscation_metrics.json',supplemental_metrics)
    lookup={(r['value'],r['column']):r for r in supplemental_metrics}
    changes=[]
    for column in supplemental.column.unique():
        clean_cell=lookup[('clean',column)]
        for attack in ('zero_width','homoglyph','base64'):
            attacked=lookup[(attack,column)]; normalized=lookup[('normalized_'+attack,column)]
            recall=lambda r:r['metrics']['recall']['value']
            fpr=lambda r:r['metrics']['false-positive rate']['value']
            changes.append({'column':column,'attack':attack,'rows':clean_cell['rows'],
                            'positive':clean_cell['positive'],'negative':clean_cell['negative'],
                            'clean_recall':recall(clean_cell),'attacked_recall':recall(attacked),
                            'normalized_recall':recall(normalized),
                            'recall_drop':recall(clean_cell)-recall(attacked),
                            'recall_recovery':recall(normalized)-recall(attacked),
                            'clean_fpr':fpr(clean_cell),'attacked_fpr':fpr(attacked),'normalized_fpr':fpr(normalized)})
    pd.DataFrame(changes).to_csv(HERE/'results'/'obfuscation_changes.csv',index=False,float_format='%.17g')
    (HERE/'checker_obfuscation_output.txt').write_text(output.getvalue(),encoding='utf-8')
    normalization=[]
    for model,part in raw.groupby('model'):
        pivot=part.pivot(index='row_id',columns='view',values='score')
        labels=original.loc[pivot.index,'label']
        threshold=float(part.threshold.iloc[0])
        normalization.append({'model':model,'clean_predictions_changed':int(((pivot.clean>=threshold)!=(pivot.normalized_clean>=threshold)).sum()),
                              'benign_clean_predictions_changed':int((((pivot.clean>=threshold)!=(pivot.normalized_clean>=threshold))&(labels==0)).sum())})
    write_json(HERE/'results'/'normalization_controls.json',normalization)
    zpath=HERE/'results'/'zeroshot_scores.csv'
    if zpath.exists():
        assemble_zero_shot()
    print('Assembled models:',list(raw.model.unique()),flush=True)
    for row in main_metrics:
        print(row['value'],row['column'],row['metrics']['recall'],row['change'],row['verdict'],flush=True)

if __name__=='__main__':
    main()
