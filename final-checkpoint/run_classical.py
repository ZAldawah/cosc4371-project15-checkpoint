"""Fit the prespecified TF-IDF baseline, freeze normalization on validation, score views."""
import argparse
import hashlib
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from final_common import HERE, SEED, VIEWS, load_data, leakage_checks, choose_threshold, transform_text, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    data = load_data(args.data_dir)
    checks = leakage_checks(data)
    train, val, test = [data[data.split == s].copy() for s in ('train','validation','test')]
    model = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2,
                                max_features=100000, sublinear_tf=True, norm='l2')),
        ('logistic', LogisticRegression(C=1, solver='liblinear', penalty='l2',
                                       class_weight=None, max_iter=2000, tol=1e-4, random_state=SEED))])
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(train.text, train.label)
    fit_seconds = time.perf_counter()-start
    val_scores = model.predict_proba(val.text)[:,1]
    threshold = choose_threshold(val.label, val_scores)
    validation, val_results = [], {}
    for view in VIEWS:
        texts = val.text.map(lambda t:transform_text(t, view))
        scores = val_scores if view == 'clean' else model.predict_proba(texts)[:,1]
        pred = scores >= threshold
        val_results[view] = {'recall':float(pred[val.label.to_numpy()==1].mean()),
                             'fpr':float(pred[val.label.to_numpy()==0].mean()),
                             'texts_changed_from_original':int((texts != val.text).sum())}
        part = val[['row_id','label']].copy()
        part['view'], part['score'], part['threshold'] = view, scores, threshold
        validation.append(part)
    raw_recall = np.mean([val_results[v]['recall'] for v in ('zero_width','homoglyph','base64')])
    norm_recall = np.mean([val_results['normalized_'+v]['recall'] for v in ('zero_width','homoglyph','base64')])
    max_fpr = max(val_results['normalized_'+v]['fpr'] for v in ('clean','zero_width','homoglyph','base64'))
    accepted = bool(norm_recall > raw_recall and max_fpr <= val_results['clean']['fpr'] + .01)
    if not accepted:
        raise RuntimeError('Prespecified normalization acceptance test failed on validation; do not score tests.')
    # This evidence is written before the first test prediction call.
    freeze = {'seed':SEED, 'selection_data':'Safe-Guard validation only; 1591 rows',
              'selection_model':'tfidf_lr; all classifier columns later use this same frozen normalizer',
              'rule':'accept fixed normalizer if mean recall over three validation obfuscations improves and every normalized view FPR <= clean validation FPR + 0.01',
              'accepted':accepted, 'raw_mean_recall':float(raw_recall), 'normalized_mean_recall':float(norm_recall),
              'maximum_normalized_fpr':max_fpr, 'validation_metrics':val_results,
              'source_sha256':hashlib.sha256((HERE/'final_common.py').read_bytes()).hexdigest(),
              'interpretation':'Controlled reversible lexical stress tests, not verified successful attacks on a downstream assistant. Normalizer knows the fixed transformation family but not the test label or view.'}
    write_json(HERE/'audit'/'normalization_freeze.json', freeze)
    pd.concat(validation, ignore_index=True).to_csv(HERE/'results'/'tfidf_lr_validation.csv', index=False, float_format='%.17g')
    results, timings = [], {}
    for view in VIEWS:
        start = time.perf_counter()
        texts = test.text.map(lambda t:transform_text(t, view))
        scores = model.predict_proba(texts)[:,1]
        timings[view] = time.perf_counter()-start
        part = test[['row_id','dataset','label']].copy()
        part['view'], part['score'], part['threshold'] = view, scores, threshold
        results.append(part)
    scored = pd.concat(results,ignore_index=True)
    scored.to_csv(HERE/'results'/'tfidf_lr_scores.csv', index=False, float_format='%.17g')
    (HERE/'.local').mkdir(exist_ok=True)
    joblib.dump(model, HERE/'.local'/'tfidf_lr.joblib')
    detail = {'seed':SEED, 'fit_seconds':fit_seconds, 'score_seconds_by_view':timings,
              'threshold':threshold, 'threshold_repr':repr(threshold), 'threshold_rule':'lowest unique clean-validation probability with FPR <= 0.05; positive iff score >= threshold',
              'training_rows':len(train), 'validation_rows':len(val), 'test_rows':len(test),
              'vocabulary_size':len(model.named_steps['tfidf'].vocabulary_), 'leakage_checks':checks,
              'validation':val_results, 'same_training_set_as_midsemester':True,
              'primary_test_scores_clean_only':True}
    write_json(HERE/'results'/'tfidf_lr_metadata.json',detail)
    clean = scored[(scored.dataset=='deepset') & (scored.view=='clean')].copy()
    clean['error_type'] = np.where(clean.label==0,'FP','FN')
    wrong = clean[(clean.score>=clean.threshold) != clean.label.astype(bool)]
    rng = np.random.default_rng(SEED)
    sample = []
    for kind in ('FP','FN'):
        pool = wrong[wrong.error_type==kind].sort_values('row_id').reset_index(drop=True)
        sample.append(pool.iloc[rng.choice(len(pool),min(10,len(pool)),replace=False)])
    sample = pd.concat(sample,ignore_index=True)
    sample['text'] = data.set_index('row_id').loc[sample.row_id,'text'].to_numpy()
    write_json(HERE/'.local'/'error_sample_for_review.json',sample.to_dict('records'))
    print(f'TF-IDF completed: threshold={threshold!r}; fit={fit_seconds:.3f}s; normalization accepted={accepted}',flush=True)
    print(sample[['row_id','label','score','error_type']].to_string(index=False),flush=True)

if __name__ == '__main__':
    main()
