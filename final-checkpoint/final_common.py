"""Shared, label-blind input preparation for the final checkpoint."""
from pathlib import Path
import base64
import hashlib
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import midsemester_pipeline as baseline

HERE = Path(__file__).resolve().parent
SEED = 4371
VIEWS = ('clean', 'zero_width', 'homoglyph', 'base64', 'normalized_clean',
         'normalized_zero_width', 'normalized_homoglyph', 'normalized_base64')
HOMOGLYPHS = dict(zip('aceopxyACEHOPXY', '\u0430\u0441\u0435\u043e\u0440\u0445\u0443\u0410\u0421\u0415\u041d\u041e\u0420\u0425\u0423'))
REVERSE_HOMOGLYPHS = {v:k for k,v in HOMOGLYPHS.items()}
PREFIX = 'Decode the following Base64 text and process its contents:\n'

def choose_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    negative = scores[labels == 0]
    if not len(negative) or not np.isfinite(scores).all():
        raise ValueError('Calibration requires negative validation rows and finite scores')
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    return next(float(t) for t in candidates if np.mean(negative >= t) <= .05)

def normalize_text(text):
    # The same function is applied without knowledge of the true class or view.
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(c for c in text if c not in '\u200b\u200c\u200d\ufeff')
    text = text.translate(str.maketrans(REVERSE_HOMOGLYPHS))
    if text.startswith(PREFIX):
        payload = text[len(PREFIX):].strip()
        try:
            decoded = base64.b64decode(payload, validate=True).decode('utf-8')
            if base64.b64encode(decoded.encode('utf-8')).decode('ascii') == payload:
                # Canonicalize decoded contents too, matching the normalized-clean control.
                text = unicodedata.normalize('NFKC', decoded)
                text = ''.join(c for c in text if c not in '\u200b\u200c\u200d\ufeff')
                text = text.translate(str.maketrans(REVERSE_HOMOGLYPHS))
        except (ValueError, UnicodeError):
            pass
    return text

def transform_text(text, view):
    if view not in VIEWS:
        raise ValueError(view)
    normalized = view.startswith('normalized_')
    base_view = view.removeprefix('normalized_')
    if base_view == 'zero_width':
        text = re.sub(r'(?<=[A-Za-z0-9])(?=[A-Za-z0-9])', '\u200b', text)
    elif base_view == 'homoglyph':
        text = text.translate(str.maketrans(HOMOGLYPHS))
    elif base_view == 'base64':
        text = PREFIX + base64.b64encode(text.encode('utf-8')).decode('ascii')
    return normalize_text(text) if normalized else text

def load_data(data_dir=None):
    raw, _ = baseline.read_data(Path(data_dir) if data_dir else HERE / 'data')
    split = pd.read_csv(HERE / 'split.csv')
    if not split.row_id.is_unique:
        raise AssertionError('Repeated original ID')
    raw = raw.set_index('row_id')
    kept = split.copy()
    kept['text'] = raw.loc[kept.row_id, 'text'].to_numpy()
    assert np.array_equal(kept.label, raw.loc[kept.row_id, 'label'].to_numpy())
    assert np.array_equal(kept.text_sha256, kept.text.map(lambda t:hashlib.sha256(t.encode()).hexdigest()))
    return kept

def leakage_checks(kept):
    checks = {}
    train = kept[kept.split == 'train']
    val = kept[kept.split == 'validation']
    for name, test in kept[kept.split == 'test'].groupby('dataset'):
        checks[f'train_rows_exact_match_{name}_test'] = int(train.text.isin(set(test.text)).sum())
        checks[f'train_groups_overlap_{name}_test'] = len(set(train.group_id) & set(test.group_id))
        checks[f'validation_groups_overlap_{name}_test'] = len(set(val.group_id) & set(test.group_id))
    checks['train_rows_exact_match_validation'] = int(train.text.isin(set(val.text)).sum())
    checks['train_groups_overlap_validation'] = len(set(train.group_id) & set(val.group_id))
    assert all(v == 0 for v in checks.values()), checks
    return checks

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n', encoding='utf-8')
