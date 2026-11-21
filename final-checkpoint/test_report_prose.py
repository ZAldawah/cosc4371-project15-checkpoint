"""Check that report interpretations follow measured or counterfactual results.

Run directly to write compact evidence to audit/report_prose_verification.json.
The tests do not train models, alter results, or rebuild the report.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest

import build_final_report as report
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
SOURCE_HASH = hashlib.sha256((HERE / 'build_final_report.py').read_bytes()).hexdigest()
MAIN = json.loads((HERE / 'results/main_metrics.json').read_text(encoding='utf-8'))
EXTRA = json.loads((HERE / 'results/obfuscation_metrics.json').read_text(encoding='utf-8'))
DISCLOSURE = ('[5] OpenAI. ChatGPT/Codex, October 29, 2026. https://chatgpt.com/. '
              'AI assistance was used for formatting and verification purposes.')
PREVIOUS_DISCLOSURE = 'AI was used in the organization, formatting, and structuring of this document.'


def indexed(rows):
    return {(r['value'], r['column']): deepcopy(r) for r in rows}


def operating_point(tp=40, fp=20, positive=100, negative=100):
    """Construct internally consistent confusion counts for a hypothetical cell."""
    return {
        'tp': tp, 'fp': fp, 'tn': negative - fp, 'fn': positive - tp,
        'positive': positive, 'negative': negative,
        'metrics': {'recall': {'value': tp / positive},
                    'false-positive rate': {'value': fp / negative}},
    }


def hypothetical_obfuscation():
    cells = {}
    for model in ('tfidf_lr', 'distilbert', 'protectai'):
        column = 'deepset__' + model
        cells[('clean', column)] = operating_point()
        for view in ('zero_width', 'homoglyph', 'base64'):
            cells[(view, column)] = operating_point(tp=50, fp=30)
            cells[('normalized_' + view, column)] = operating_point()
    cells[('clean', 'safeguard__distilbert')] = operating_point()
    cells[('normalized_clean', 'safeguard__distilbert')] = operating_point()
    return cells


def hypothetical_verdicts():
    rows = deepcopy(MAIN)
    for row in rows:
        if row['value'] != 'safeguard':
            row['verdict'], row['change'] = 'no', 0.01
    return rows


class CurrentResults(unittest.TestCase):
    def test_measured_false_alarm_tradeoff_is_visible(self):
        text = report.false_alarm_note(indexed(MAIN))
        self.assertIn('57.79% (230/398)', text)
        self.assertIn('6.03% (24/398)', text)
        self.assertIn('4.77% (19/398)', text)
        self.assertIn("DistilBERT's higher recall", text)
        self.assertIn('higher false-positive rate', text)

    def test_measured_loss_and_supported_cells(self):
        text = report.loss_finding(indexed(MAIN)[('deepset', 'tfidf_lr')])
        self.assertIn('0.5861 (58.61 percentage points)', text)
        self.assertIn('meets the prespecified material-loss rule', text)
        claim = report.claim_text(MAIN)
        for name in ('TF-IDF + LR', 'DistilBERT', 'ProtectAI v2'):
            self.assertIn('deepset / ' + name, claim)
        self.assertNotIn('Gandalf', claim)

    def test_measured_normalization_and_false_alarm_counts(self):
        text = report.obfuscation_interpretation(indexed(EXTRA))
        self.assertIn('5 of 6 transformer obfuscation cells', text)
        self.assertIn('FPR 0.9673-1.0000', text)
        self.assertIn('9 of 9 normalized cells', text)
        self.assertIn('increases false alarms (82 to 83 out of 1,409', text)


class CounterfactualFalseAlarms(unittest.TestCase):
    def test_tradeoff_claim_disappears_unless_both_metrics_increase(self):
        # TP/100 and FP/100 represent recall and FPR. LR is fixed at 60/20.
        cases = [(50, 30, False), (60, 30, False), (70, 10, False),
                 (70, 20, False), (60, 20, False), (70, 30, True)]
        for tp, fp, tradeoff in cases:
            with self.subTest(distilbert_tp=tp, distilbert_fp=fp):
                cells = {('deepset', 'tfidf_lr'): operating_point(tp=60, fp=20),
                         ('deepset', 'distilbert'): operating_point(tp=tp, fp=fp),
                         ('deepset', 'protectai'): operating_point(tp=65, fp=15)}
                text = report.false_alarm_note(cells)
                self.assertEqual("DistilBERT's higher recall" in text, tradeoff)
                self.assertEqual('higher false-positive rate' in text, tradeoff)
                self.assertIn(f'({fp}/100)', text)


class CounterfactualVerdicts(unittest.TestCase):
    def test_loss_finding_does_not_always_claim_material_loss(self):
        cases = [(.10, 'yes', 'meets the prespecified material-loss rule'),
                 (.0799, 'no', 'meets the prespecified no-material-loss rule'),
                 (.07995, 'undecided', 'falls in the prespecified undecided gap'),
                 (0.0, 'no', 'meets the prespecified no-material-loss rule'),
                 (-.12, 'no', 'meets the prespecified no-material-loss rule')]
        for change, verdict, interpretation in cases:
            with self.subTest(change=change, verdict=verdict):
                text = report.loss_finding({'change': change, 'verdict': verdict})
                self.assertIn(interpretation, text)
                self.assertNotIn('exceeding the prespecified 8-point', text)
        self.assertIn('-12.00 percentage points',
                      report.loss_finding({'change': -.12, 'verdict': 'no'}))

    def test_mixed_yes_no_undecided_names_only_supported_losses(self):
        rows = hypothetical_verdicts()
        selected = {(r['value'], r['column']): r for r in rows}
        selected[('gandalf', 'protectai')].update(verdict='yes', change=.2)
        selected[('deepset', 'distilbert')].update(verdict='undecided', change=.07995)
        text = report.claim_text(rows)
        self.assertIn('Gandalf* / ProtectAI v2 (D=0.2000)', text)
        self.assertNotIn('deepset / TF-IDF + LR', text)
        self.assertNotIn('deepset / DistilBERT', text)
        self.assertIn('1 external cell(s) remain inconclusive', text)
        self.assertNotIn('All measured external cells meet', text)

    def test_no_and_undecided_do_not_become_all_no(self):
        rows = hypothetical_verdicts()
        next(r for r in rows if r['value'] == 'deepset').update(
            verdict='undecided', change=.07995)
        text = report.claim_text(rows)
        self.assertIn('1 external cell(s) remain inconclusive', text)
        self.assertNotIn('All measured external cells meet', text)
        self.assertNotIn('is supported for', text)

    def test_all_no_does_not_claim_an_inconclusive_or_positive_result(self):
        text = report.claim_text(hypothetical_verdicts())
        self.assertIn('All measured external cells meet the prespecified no-material-loss rule', text)
        self.assertNotIn('inconclusive', text.lower())
        self.assertNotIn('is supported for', text)

    def test_all_undecided_uses_the_handout_required_claim(self):
        rows = hypothetical_verdicts()
        for row in rows:
            if row['value'] != 'safeguard':
                row.update(verdict='undecided', change=.07995)
        self.assertEqual(report.claim_text(rows),
                         'Inconclusive under the prespecified decision rule.')


class CounterfactualNormalization(unittest.TestCase):
    def test_false_alarm_collapse_claim_disappears_when_absent(self):
        text = report.obfuscation_interpretation(hypothetical_obfuscation())
        self.assertIn('No transformer obfuscation cell has FPR at least 95%', text)
        self.assertNotIn('5 of 6', text)
        self.assertNotIn('recall reaches 1', text)

    def test_high_fpr_count_and_recall_range_follow_changed_cells(self):
        cells = hypothetical_obfuscation()
        cells[('homoglyph', 'deepset__distilbert')] = operating_point(tp=20, fp=95)
        cells[('base64', 'deepset__protectai')] = operating_point(tp=90, fp=96)
        cells[('zero_width', 'deepset__protectai')] = operating_point(tp=100, fp=94)
        text = report.obfuscation_interpretation(cells)
        self.assertIn('2 of 6 transformer obfuscation cells', text)
        self.assertIn('FPR 0.9500-0.9600; recall 0.2000-0.9000', text)

    def test_partial_restoration_cannot_be_described_as_complete(self):
        cells = hypothetical_obfuscation()
        cells[('normalized_base64', 'deepset__tfidf_lr')] = operating_point(tp=39, fp=20)
        cells[('normalized_homoglyph', 'deepset__protectai')] = operating_point(tp=40, fp=21)
        text = report.obfuscation_interpretation(cells)
        self.assertIn('7 of 9 normalized cells match their clean recall and FPR', text)
        self.assertNotIn('9 of 9 normalized cells', text)
        self.assertNotIn('normalization restores the clean operating point', text.lower())

    def test_clean_control_false_alarms_may_increase_decrease_or_stay_equal(self):
        for after, phrase in [(21, 'increases'), (19, 'decreases'), (20, 'leaves unchanged')]:
            with self.subTest(normalized_fp=after):
                cells = hypothetical_obfuscation()
                cells[('normalized_clean', 'safeguard__distilbert')] = operating_point(fp=after)
                text = report.obfuscation_interpretation(cells)
                self.assertIn(f'{phrase} false alarms (20 to {after} out of 100', text)
                self.assertNotIn('82 to 83', text)

    def test_helpers_do_not_mutate_input_results(self):
        main, extra = indexed(MAIN), indexed(EXTRA)
        original_main, original_extra = deepcopy(main), deepcopy(extra)
        report.false_alarm_note(main)
        report.loss_finding(main[('deepset', 'tfidf_lr')])
        report.claim_text(list(main.values()))
        report.obfuscation_interpretation(extra)
        self.assertEqual(main, original_main)
        self.assertEqual(extra, original_extra)

    def test_requested_disclosure_is_preserved(self):
        builder = (HERE / 'build_final_report.py').read_text(encoding='utf-8')
        readme = (HERE / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(report.AI_REFERENCE, DISCLOSURE)
        self.assertIn(DISCLOSURE, readme)
        self.assertNotIn(PREVIOUS_DISCLOSURE, builder)
        self.assertNotIn(PREVIOUS_DISCLOSURE, readme)
        pages = PdfReader(HERE / 'Aldawah_COSC4371_Final_Checkpoint.pdf').pages
        pdf_text = ' '.join(' '.join(page.extract_text() or '' for page in pages).split())
        self.assertEqual(pdf_text.count(DISCLOSURE), 1)
        self.assertNotIn(PREVIOUS_DISCLOSURE, pdf_text)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        'status': 'passed' if result.wasSuccessful() else 'failed',
        'command': 'python test_report_prose.py',
        'python': platform.python_version(),
        'tests_run': result.testsRun,
        'failed_tests': [case.id() for case, _ in result.failures],
        'error_tests': [case.id() for case, _ in result.errors],
        'report_source_sha256': SOURCE_HASH,
        'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'result_fixture_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                  for name in ('results/main_metrics.json', 'results/obfuscation_metrics.json')},
        'scope': 'Measured result interpretations and counterfactual metric/verdict changes; no model reruns or result modifications',
    }
    (HERE / 'audit' / 'report_prose_verification.json').write_text(
        json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
