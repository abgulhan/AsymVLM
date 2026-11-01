"""Strict coverage reporting with Phi's frozen singular EOS trace schema.

This is a reporting-only compatibility repair. Physical evidence, stopping
rules, original source pins, comparison cohorts and scientific methods are not
modified. The previous report and any previous reporting error remain intact.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading

from . import coverage_audit_resume_report as previous
from .state import atomic_json, file_digest

FORMAT = 'mlsys_coverage_audit_resume_strict_report_v2'
PREVIOUS_SHA256 = 'e69e1904acaace729d0e2e50d5ef909135fa5ac401b019404d5133d5cff945be'
_VALIDATE_ROW = previous.validate_row
_LOCK = threading.RLock()


def effective_eos(row):
    """Accept the two schemas actually emitted by the pinned generation code."""
    metrics = row['metrics']
    if row['model'] == 'phi35':
        value = metrics.get('effective_eos_token_id')
        stopping = metrics.get('stopping', {})
        if (type(value) is not int or value != 32000
                or stopping.get('policy') != 'phi_tokenizer_eos_unpenalized_greedy_v4'
                or stopping.get('effective_eos_token_id') != value
                or stopping.get('tokenizer_eos_token_id') != value
                or stopping.get('max_new_tokens_is_only_a_safety_cap') is not True):
            raise ValueError('Phi EOS must match the frozen tokenizer stopping receipt')
        eos = [value]
        if ('effective_eos_token_ids' in metrics
                and metrics['effective_eos_token_ids'] != eos):
            raise ValueError('conflicting singular/plural EOS trace')
    else:
        eos = metrics.get('effective_eos_token_ids')
        if 'effective_eos_token_id' in metrics:
            raise ValueError('singular EOS trace is only defined for the Phi adapter')
    if (not isinstance(eos, list)
            or any(type(token) is not int or token < 0 for token in eos)
            or len(set(eos)) != len(eos)):
        raise ValueError('invalid effective EOS token list')
    return eos


def validate_row(row, task, entry, method):
    if method.startswith('coverage_greedy'):
        elapsed = previous.finite(row.get('whole_selector_elapsed_seconds'),
                                  'whole selector elapsed seconds')
        if (row.get('whole_selector_deadline_checked') is not True
                or row.get('whole_selector_limit_seconds') != 60.0
                or (row.get('status') == 'ok' and elapsed > 60.0)):
            raise ValueError('greedy result lacks the frozen whole-selector deadline proof')
    if row.get('status') != 'ok':
        return _VALIDATE_ROW(row, task, entry, method)
    eos = effective_eos(row)
    # Copy only the outer dictionaries; no input evidence or token arrays change.
    normalized = dict(row, metrics=dict(row['metrics'], effective_eos_token_ids=eos))
    _VALIDATE_ROW(normalized, task, entry, method)
    metrics = row['metrics']
    tokens = metrics['generated_token_ids']
    ended = tokens[-1] in eos
    if (metrics.get('generated_tokens') != len(tokens)
            or metrics.get('ended_with_eos') is not ended
            or metrics.get('stop_reason') != ('eos_token' if ended else 'max_new_tokens')
            or metrics.get('first_eos_position') != (len(tokens)-1 if ended else None)):
        raise ValueError('natural-stopping annotations contradict the generated token trace')


@contextmanager
def validation_binding():
    with _LOCK:
        if file_digest(previous.__file__) != PREVIOUS_SHA256:
            raise ValueError('frozen previous strict reporter changed')
        if previous.validate_row is not _VALIDATE_ROW:
            raise ValueError('strict row validator already has an unexpected binding')
        previous.validate_row = validate_row
        try:
            yield
        finally:
            previous.validate_row = _VALIDATE_ROW


def collect(root):
    with validation_binding():
        result = previous.collect(root)
    pins = result[-1]
    pins.add(previous.__file__, PREVIOUS_SHA256)
    pins.add(__file__)
    return result


def report(root):
    audit, expected, rows, pending, failures, shards, cpu, pins = collect(root)
    fixed, adaptive, cohorts = previous.tables(expected, rows)
    result = {'format': FORMAT, 'audit_sha256': audit['sha256'], 'complete': not pending,
        'status': 'pending' if pending else 'finished_with_failures' if failures else 'complete',
        'scientific_success_claimed': False, 'independent_confirmation': False,
        'expected_physical_cells': len(expected)*len(previous.original.METHODS),
        'validated_physical_cells': len(rows), 'pending_shards': pending,
        'completed_shards': shards, 'physical_failures': failures, 'cohorts': cohorts,
        'fixed_budget': fixed, 'adaptive_compression': adaptive, 'cpu_reuse': cpu,
        'mme_metric': 'complete-pair question accuracy diagnostic; not official MME category macro/total',
        'latency_claim': 'live selector included; shared-node development diagnostic, no isolated speedup claim',
        'completed_cohorts_only': True, 'input_hashes': pins.values,
        'reporter_sha256': file_digest(__file__),
        'reporting_repair': {'previous_reporter_sha256': PREVIOUS_SHA256,
            'reason': 'Phi emits effective_eos_token_id, not the plural field used by Qwen/LLaVA',
            'normalization_scope': 'validated in-memory row copy only',
            'original_evidence_unchanged': True, 'stopping_policy_changed': False,
            'whole_selector_deadline_verified': True}}
    atomic_json(Path(root)/'report-v2.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    value = report(args.root)
    print(json.dumps({key: value[key] for key in
        ('status', 'complete', 'expected_physical_cells', 'validated_physical_cells')}))


if __name__ == '__main__':
    main()
