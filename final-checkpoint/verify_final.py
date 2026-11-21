"""Independent artifact checks beyond the instructor's structural checker."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pypdf import PdfReader
from final_common import HERE, SEED, VIEWS, choose_threshold, write_json

def main():
    checks={}
    split=pd.read_csv(HERE/'split.csv')
    test=split[split.split=='test'].set_index('row_id')
    main=pd.read_csv(HERE/'scores.csv')
    assert len(main)==len(test)*3 and not main.duplicated(['row_id','value','column']).any()
    assert set(main.column)=={'tfidf_lr','distilbert','protectai'}
    for (value,column),cell in main.groupby(['value','column']):
        expected=set(test[test.dataset==value].index)
        assert set(cell.row_id)==expected
        assert np.array_equal(cell.label,test.loc[cell.row_id,'label'])
        assert np.isfinite(cell[['score','threshold']]).all().all()
    checks['main_cells_complete']=9
    checks['main_score_rows']=len(main)
    supplementary=pd.read_csv(HERE/'scores_obfuscation.csv')
    expected_columns={pool+'__'+model for pool in ['safeguard','deepset','gandalf','gandalf_mixed'] for model in ['tfidf_lr','distilbert','protectai']}
    assert set(supplementary.column)==expected_columns
    assert len(supplementary)==146976
    assert not supplementary.duplicated(['row_id','value','column']).any()
    for column,group in supplementary.groupby('column'):
        pool=column.split('__')[0]
        expected_ids=set(test[(test.dataset=='gandalf')|((test.dataset=='safeguard')&(test.label==0))].index) if pool=='gandalf_mixed' else set(test[test.dataset==pool].index)
        assert set(group.value)==set(VIEWS)
        sets=[set(cell.row_id) for _,cell in group.groupby('value')]
        assert all(s==sets[0] for s in sets)
        assert sets[0]==expected_ids
        assert group.threshold.nunique()==1
        assert np.array_equal(group.label,test.loc[group.row_id,'label'])
    checks['supplemental_cells_complete']=int(supplementary.groupby(['value','column']).ngroups)
    assert checks['supplemental_cells_complete']==96
    checks['supplemental_score_rows']=len(supplementary)
    for model in ['tfidf_lr','distilbert','protectai']:
        raw=pd.read_csv(HERE/'results'/f'{model}_scores.csv',float_precision='round_trip')
        default=pd.read_csv(HERE/'results'/f'{model}_scores.csv')
        assert np.array_equal(raw.score>=raw.threshold,default.score>=default.threshold),model
        vpath=HERE/'results'/('tfidf_lr_validation.csv' if model=='tfidf_lr' else f'{model}_validation_scores.csv')
        val=pd.read_csv(vpath,float_precision='round_trip')
        if 'view' in val: val=val[val['view']=='clean']
        assert set(val.row_id)==set(split[split.split=='validation'].row_id)
        t=choose_threshold(val.label,val.score)
        assert np.all(raw.threshold.to_numpy()==t),model
        assert (val.loc[val.label==0,'score']>=t).mean()<=.05
        if model!='tfidf_lr':
            audit=json.loads((HERE/'audit'/f'transformer_{model}.json').read_text())
            assert audit['status']=='complete'
            assert audit['split_sha256']==hashlib.sha256((HERE/'split.csv').read_bytes()).hexdigest()
        checks[model+'_threshold_matches_validation']=True
    errors=pd.read_csv(HERE/'errors.csv')
    cell=main[(main.value=='deepset')&(main.column=='tfidf_lr')].copy()
    wrong=cell[(cell.score>=cell.threshold)!=cell.label.astype(bool)].copy()
    rng=np.random.default_rng(SEED); expected=[]
    for label in [0,1]:
        pool=wrong[wrong.label==label].sort_values('row_id').reset_index(drop=True)
        expected.extend(pool.iloc[rng.choice(len(pool),min(10,len(pool)),replace=False)].row_id)
    assert list(errors.row_id)==expected
    assert errors.category.notna().all() and len(errors)==20
    checks['error_sample_reproduced']=True
    summaries=json.loads((HERE/'results'/'main_metrics.json').read_text())
    output=(HERE/'checker_output.txt').read_text()
    for row in summaries:
        metric=row['metrics']['recall']; lo,hi=metric['ci95']
        assert f"{metric['value']:.4f} [{lo:.4f}, {hi:.4f}]" in output
        if row['value']!='safeguard': assert f"change {round(row['change'],4)+0.0:+.4f}" in output
    checks['summary_metrics_match_checker_text']=True
    pdf=PdfReader(HERE/'Aldawah_COSC4371_Final_Checkpoint.pdf')
    text='\n'.join(page.extract_text() or '' for page in pdf.pages)
    for heading in ['1. Main comparison','2. Figure','3. Claim and limitation','4. Error analysis','5. Reproducibility','6. Plan to finish','Appendix B. Full primary-checker output']:
        assert heading in text,heading
    assert REPO in text if (REPO:='https://github.com/ZAldawah/cosc4371-project15-checkpoint') else False
    assert 'Zainulabdeen Aldawah' in (pdf.pages[0].extract_text() or '')
    body=' '.join(text.split('Appendix A.',1)[0].split())
    for row in summaries:
        metric=row['metrics']['recall']; lo,hi=metric['ci95']
        assert f"{metric['value']:.4f} [{lo:.4f}, {hi:.4f}]" in body
        if row['value']!='safeguard': assert f"{row['change']:+.4f}" in body
    checks['pdf_primary_numbers_match_summary']=True
    # Line wrapping is allowed in the appendix; every output word must remain in sequence.
    appendix=text.split('Appendix B. Full primary-checker output',1)[1]
    pos=0
    for word in output.split():
        found=appendix.find(word,pos)
        assert found>=0, f'Missing checker output token: {word}'
        pos=found+len(word)
    checks['full_checker_output_present_in_pdf']=True
    checks['pdf_pages']=len(pdf.pages)
    freeze=json.loads((HERE/'audit'/'normalization_freeze.json').read_text())
    assert freeze['accepted'] and freeze['source_sha256']==hashlib.sha256((HERE/'final_common.py').read_bytes()).hexdigest()
    checks['normalization_freeze_matches_source']=True
    data=json.loads((HERE/'audit'/'final_data_verification.json').read_text())
    assert data['split_matches_midsemester'] and all(v==0 for v in data['leakage_checks'].values())
    checks['full_data_audit_passed']=True
    zero=pd.read_csv(HERE/'scores_zeroshot.csv')
    assert set(zero.row_id)==set(test.index) and zero.row_id.is_unique
    assert (zero.threshold==0).all()
    assert np.isfinite(zero[['score','threshold']]).all().all()
    assert np.array_equal(zero.label,test.loc[zero.row_id,'label'])
    assert np.array_equal(zero.value,test.loc[zero.row_id,'dataset'])
    checks['strict_zero_shot_complete']=True
    write_json(HERE/'audit'/'final_verification.json',{'status':'passed','checks':checks})
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__': main()
