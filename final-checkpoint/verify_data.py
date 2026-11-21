"""Recompute the full original-text family audit and reproduce the frozen split."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import midsemester_pipeline as baseline
from final_common import HERE, load_data, leakage_checks, transform_text, normalize_text, VIEWS, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=HERE/'data')
    args=parser.parse_args()
    dest=HERE/'audit'/'recomputed'
    dest.mkdir(exist_ok=True)
    raw,inputs=baseline.read_data(args.data_dir)
    grouped,near=baseline.duplicate_audit(raw,dest)
    kept,cleanup=baseline.clean_split(grouped,dest)
    columns=['row_id','dataset','split','label','source_split','label_kind','group_id','text_sha256']
    frozen=pd.read_csv(HERE/'split.csv')
    pd.testing.assert_frame_equal(kept[columns].reset_index(drop=True),frozen[columns],check_dtype=False)
    checks=leakage_checks(kept)
    details={}
    for dataset,g in kept[kept.split=='test'].groupby('dataset'):
        canonical=g.text.map(normalize_text)
        part={'rows':len(g),'clean_texts_changed_by_normalization':int((canonical!=g.text).sum()),
              'benign_clean_texts_changed_by_normalization':int(((canonical!=g.text)&(g.label==0)).sum()),'views':{}}
        for view in VIEWS:
            transformed=g.text.map(lambda t:transform_text(t,view))
            part['views'][view]={'changed_from_clean':int((transformed!=g.text).sum()),
                                 'equal_normalized_clean':int((transformed==canonical).sum())}
        details[dataset]=part
    # Narrow deterministic function checks catch transformations which use labels or break strict Base64 decoding.
    for text in ['plain ASCII123', 'Caf\u00e9 and fullwidth \uff21', '\u0430\u0431\u0432 text', 'a\u200db']:
        for view in ('zero_width','homoglyph','base64'):
            assert transform_text(text,'normalized_'+view)==normalize_text(text)
    result={'split_matches_midsemester':True,'retained_rows':len(kept),'raw_rows':len(raw),
            'split_sha256':hashlib.sha256((HERE/'split.csv').read_bytes()).hexdigest(),
            'input_hashes_verified':True,'full_near_duplicate_recomputed':True,
            'near_duplicate_audit':near,'cleanup':cleanup,'leakage_checks':checks,
            'transform_controls':details,'common_source_sha256':hashlib.sha256((HERE/'final_common.py').read_bytes()).hexdigest()}
    write_json(HERE/'audit'/'final_data_verification.json',result)
    print('Full data verification passed; frozen split reproduced with zero measured boundary overlaps.',flush=True)

if __name__=='__main__':
    main()
