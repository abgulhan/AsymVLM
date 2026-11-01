"""Bounded identity-only re-audit and full available evaluation admission.

Reuse, but never rewrite, the v9 source inventory and historical receipts.
No question, answer, prediction, or pickle payload is loaded. Unresolved
provenance blocks independent confirmation, rather than being guessed from a
filename. Each admitted manifest contains every eligible available evaluation
row, not a favorable subset or another 1,000-example sample.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

from learned_score.mlsys_eval import historical_context, historical_resolver, protocol
from learned_score.mlsys_eval.common import load_manifest, seal_manifest
from . import compression_data_v9 as old
from . import compression_features_v12 as features
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_identity_admission_v12'
MAX_SECONDS = 2 * 3600


def _check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError('identity admission reached its two-hour/study deadline')


def bounded_hash(path, deadline):
    _check_deadline(deadline)
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            _check_deadline(deadline)
            value.update(block)
    return value.hexdigest()


def revalidate_pins(pins, deadline):
    for path, expected in pins.items():
        if bounded_hash(path, deadline) != expected:
            raise ValueError('historical source bytes changed: '+str(path))


def resolve_issue(issue, *, deadline, content_receipts=None):
    """Try content receipts, explicit context, and safe JSON identity fields."""
    _check_deadline(deadline)
    path = Path(issue['path'])
    actual = bounded_hash(path, deadline)
    if actual != issue['sha256']:
        raise ValueError('historical issue changed since the original audit')
    common = {'artifact_path': str(path), 'artifact_sha256': actual, 'labels_loaded': False,
              'predictions_read': False, 'binary_deserialization': False}
    receipt = (content_receipts or {}).get(str(path))
    if receipt:
        old.checked(receipt)
        if (receipt.get('artifact_sha256') != actual or receipt.get('resolved') is not True
                or receipt.get('proven_dataset') != 'MME'
                or receipt.get('proof_kind') not in ('all_record_identity_join', 'byte_identical_pinned_source')):
            raise ValueError('invalid historical content-identity receipt')
        if bounded_hash(receipt['source_path'], deadline) != receipt['source_sha256']:
            raise ValueError('content-identity source changed')
        return old.seal({**common, 'resolved': True, 'proven_dataset': 'MME',
                         'proof_kind': 'revalidated_content_receipt', 'receipt_sha256': receipt['sha256'],
                         'possible_domains': ['MME'], 'entire_source_previously_examined': True,
                         'identity_keys': []})
    # Existing context helper reads only recognized dataset metadata lines.
    names, evidence = old._metadata_scope(path)
    names = sorted(set(names))
    scope = [name for name in names if name in old.FAMILIES]
    out_of_scope = {old.family(name) for name in historical_resolver.OUT_OF_SCOPE} - set(old.FAMILIES)
    if names and not scope and all(name in out_of_scope for name in names):
        return old.seal({**common, 'resolved': True, 'proof_kind': 'explicit_out_of_scope_metadata',
                         'possible_domains': [], 'identity_keys': [], 'metadata_evidence': evidence})
    identity_keys, projected = set(), []
    if path.suffix in ('.json', '.jsonl'):
        projected = list(historical_resolver.iter_identity_records(path))
        for record in projected:
            _check_deadline(deadline)
            domain = old.family(record.get('dataset', record.get('dataset_name', record.get('source_dataset'))))
            if domain not in old.FAMILIES and len(scope) == 1:
                domain = scope[0]
            keys = old.qualify_keys(record, proven_family=domain if domain in old.FAMILIES else None)
            if not keys:
                break
            identity_keys.update(keys)
        else:
            if projected and identity_keys:
                return old.seal({**common, 'resolved': True, 'proof_kind': 'explicit_identity_fields',
                    'possible_domains': scope or issue['possible_domains'], 'identity_keys': sorted(identity_keys),
                    'identity_records': len(projected), 'identity_projection_sha256': digest(projected),
                    'metadata_evidence': evidence})
    return old.seal({**common, 'resolved': False,
        'reason': 'binary_identity_projection_unavailable' if path.suffix in ('.pkl', '.pickle')
                  else 'no_complete_source_bound_identity_projection',
        'possible_domains': scope or issue['possible_domains'], 'metadata_evidence': evidence,
        'identity_records': len(projected), 'identity_projection_sha256': digest(projected),
        'identity_keys': [], 'filename_scope_inference_used': False})


def evaluation_inventory(root, prior_status, *, deadline, historical_pins=None):
    """Read all available official validation/test IDs from v9's source list."""
    import pyarrow.parquet as pq
    rows, sources, offsets, blockers = [], [], defaultdict(int), defaultdict(list)
    downloaded = {}
    download_path = root/'reproduction_results/native_protocol/download_inventory.json'
    if str(download_path) in (historical_pins or {}):
        if bounded_hash(download_path, deadline) != historical_pins[str(download_path)]:
            raise ValueError('original download inventory changed')
        inventory = json.loads(download_path.read_text())
        for dataset in inventory['datasets'].values():
            for record in dataset.get('files', []):
                path = Path(record['path'])
                path = path if path.is_absolute() else root/path
                downloaded[str(path.resolve())] = {'sha256': record['sha256'],
                    'dataset': dataset['repo'], 'revision': dataset['revision']}
    for record in prior_status['source_files']:
        _check_deadline(deadline)
        path = Path(record['path'])
        path = path if path.is_absolute() else root/path
        if path.suffix == '.parquet' and path.name.startswith('validation-'):
            if 'textvqa' in str(path):
                family, dataset, revision = 'TextVQA', old.TEXT_REPO, old.TEXT_REV
                columns = ['question_id', 'image_id']
            elif 'docvqa' in str(path):
                family, dataset, revision = 'DocVQA', old.DOC_REPO, old.DOC_REV
                columns = ['question_id', 'other_metadata']
            else:
                continue
            actual = bounded_hash(path, deadline)
            expected = record.get('sha256')
            recovery = downloaded.get(str(path.resolve()))
            if expected is None and recovery:
                if recovery['dataset'] != dataset or recovery['revision'] != revision:
                    raise ValueError('download inventory dataset/revision differs')
                expected = recovery['sha256']
            if expected is None:
                blockers[family].append({'code': 'historical_source_hash_missing', 'path': str(path),
                                         'observed_sha256': actual})
            elif actual != expected:
                blockers[family].append({'code': 'historical_source_hash_changed', 'path': str(path)})
                continue
            raw_rows = pq.read_table(path, columns=columns).to_pylist()
            for index, raw in enumerate(raw_rows):
                meta = raw.get('other_metadata') or {}
                image = str(raw.get('image_id') or meta.get('image') or '')
                document = str(meta.get('ucsf_document_id', ''))
                if not image or (family == 'DocVQA' and not document):
                    raise ValueError('source lacks image/document identity')
                row = {'dataset': dataset, 'dataset_revision': revision, 'source_split': 'validation',
                    'dataset_index': offsets[family]+index, 'question_id': str(raw['question_id']),
                    'image_id': image, 'group_id': f'{family}:document:{document}' if document else f'{family}:image:{image}',
                    'group_provenance': 'ucsf_document_id' if document else 'image_id',
                    'source_format': 'parquet', 'source_file': str(path.resolve()), 'source_row_index': index,
                    'source_shard_sha256': actual}
                if document:
                    row['document_id'] = document
                row['row_hash'] = digest(row)
                rows.append(row)
            offsets[family] += len(raw_rows)
            sources.append({'path': str(path), 'sha256': actual, 'dataset': family,
                            'sha256_provenance': 'v9_source_inventory' if record.get('sha256') else str(download_path),
                            'rows': len(raw_rows), 'columns_read': columns})
        elif path.suffix == '.tsv' and path.stem in ('MME', 'ChartQA_TEST'):
            domain = path.stem
            actual = bounded_hash(path, deadline)
            if actual != record.get('sha256'):
                blockers[domain].append({'code': 'historical_source_hash_missing_or_changed', 'path': str(path)})
                continue
            added = old.data._identity_tsv(path, domain)
            rows.extend(added)
            sources.append({'path': str(path), 'sha256': actual, 'dataset': domain, 'rows': len(added),
                            'columns_read': ['index', 'image_path'] + (['category'] if domain == 'MME' else [])})
    old.identity_only(rows)
    for domain in old.FAMILIES:
        if not any(old.family(row['dataset']) == domain for row in rows):
            blockers[domain].append({'code': 'no_available_official_evaluation_source'})
    return rows, sources, dict(blockers)


def assess(rows, exclusions, history_keys, receipts, source_blockers=None):
    """Pure whole-image/document exclusion and separate opened-data accounting."""
    old.identity_only(rows)
    old.identity_only(exclusions)
    fixed_keys = set()
    for row in exclusions:
        fixed_keys.update(old.qualify_keys(row))
    training_free, fixed_groups = old.group_closure(rows, fixed_keys)
    historical = set(history_keys)
    for receipt in receipts:
        historical.update(receipt.get('identity_keys', []))
    available, prior_groups = old.group_closure(training_free, historical)
    whole_examined = {r['proven_dataset'] for r in receipts if r.get('entire_source_previously_examined')}
    domains = {}
    for domain in old.FAMILIES:
        source = [r for r in rows if old.family(r['dataset']) == domain]
        before_history = [r for r in training_free if old.family(r['dataset']) == domain]
        selected = [r for r in available if old.family(r['dataset']) == domain]
        unresolved = [r for r in receipts if not r['resolved'] and domain in r['possible_domains']]
        reasons = copy.deepcopy((source_blockers or {}).get(domain, []))
        if unresolved:
            reasons.append({'code': 'unresolved_historical_identity', 'artifacts': len(unresolved)})
        if domain in whole_examined or domain == 'MME':
            reasons.append({'code': 'official_evaluation_source_previously_examined',
                            'scope': 'entire_dataset', 'evidence_class': 'development_or_transfer_only'})
        if not selected:
            reasons.append({'code': 'no_independent_rows_after_exclusions'})
        domains[domain] = {'dataset': domain, 'eligible': not reasons,
            'status': 'eligible_pending_image_verification' if not reasons else 'blocked',
            'available_official_rows': len(source), 'training_calibration_excluded_rows': len(source)-len(before_history),
            'training_calibration_excluded_groups': sum(g[0] == domain for g in fixed_groups),
            'previously_examined_rows': len(before_history) if domain in whole_examined else len(before_history)-len(selected),
            'provisional_independent_rows': len(selected), 'provisional_independent_groups': len({r['group_id'] for r in selected}),
            'complete_available_split': True, 'sample_cap': None, 'blocking_reasons': reasons,
            'unresolved_artifacts': [r['artifact_path'] for r in unresolved], 'labels_loaded': False}
    return domains, available


def full_suite_readiness(study, admission=None):
    """Explicit publication barrier; no fake jobs or independent-data claims."""
    reasons = []
    if admission is None:
        reasons.append({'code': 'awaiting_identity_audit'})
    elif admission.get('study_sha256') != study['sha256'] or admission.get('labels_loaded') is not False:
        raise ValueError('full-suite gate received invalid admission')
    elif not admission.get('manifests'):
        reasons.append({'code': 'identity_admission_blocked',
                        'domains': {k: v['blocking_reasons'] for k, v in admission['domains'].items()}})
    else:
        reasons.append({'code': 'awaiting_frozen_deployment_and_full_suite_gate',
                        'detail': 'compression_full_suite_v12.advance binds eligible manifests and deployment before publishing jobs.'})
    return old.seal({'format': FORMAT+'_full_suite_readiness', 'study_sha256': study['sha256'],
        'status': 'blocked', 'ready': False, 'plans_published': False, 'labels_opened': False,
        'complete_available_evaluation_split_required': True, 'blocking_reasons': reasons,
        'admission_sha256': admission.get('sha256') if admission else None,
        'runtime_binding': 'compression_full_suite_v12.advance'})


def publish(study):
    source = features.verify_study(study)
    root, output = Path(study['cwd']), Path(study['output_root'])/'admission'
    if (output/'status.json').exists():
        value = old.checked(json.loads((output/'status.json').read_text()))
        if value.get('study_sha256') != study['sha256'] or value.get('labels_loaded') is not False:
            raise ValueError('existing admission belongs to another study or opened labels')
        return value
    started = time.time()
    deadline = min(study['deadline_utc_epoch'], started+MAX_SECONDS,
                   study.get('identity_audit_deadline_utc_epoch', started+MAX_SECONDS))
    status_path = root/old.DEFAULT_ROOT/'data/status.json'
    prior = old.checked(json.loads(status_path.read_text()))
    history_path = root/old.DEFAULT_ROOT/'data/audits'/(prior['history_sha256']+'.json')
    history = old.checked(json.loads(history_path.read_text()))
    triage_path = root/old.DEFAULT_ROOT/'data/triage/provenance_content_v1.json'
    triage = old.checked(json.loads(triage_path.read_text()))
    receipts, pins, manifests = [], {}, {}
    partial_error = None
    source_files, rows, exclusions, domain_sources = [], [], [], {}
    try:
        # Every source supporting the old exclusion keys is revalidated, not
        # only the convenient unresolved subset or an old status document.
        revalidate_pins(history['input_hashes'], deadline)
        pins.update(history['input_hashes'])
        revalidate_pins(triage['input_hashes'], deadline)
        pins.update(triage['input_hashes'])
        proven = {r['artifact_path']: r for r in triage['resolved_receipts']}
        for issue in history['issues']:
            receipt = resolve_issue(issue, deadline=deadline, content_receipts=proven)
            receipts.append(receipt)
            write_immutable(output/'receipts'/(digest(issue['path'])+'.json'), receipt)
        rows, source_files, domain_sources = evaluation_inventory(root, prior, deadline=deadline,
                                                                 historical_pins=history['input_hashes'])
        exclusions = old.frozen_exclusions(root)
        domains, available = assess(rows, exclusions, history['identity_keys'], receipts, domain_sources)
        # All v11 development identities are already covered by the v4 manifest
        # exclusion. Do not open labels, or materialize annotation TSVs here.
        old_images = {r['image_sha256'] for r in exclusions if r.get('image_sha256')}
        old_images.update(k.split(':', 1)[1] for k in history['identity_keys'] if k.startswith('image_sha256:'))
        for domain, status in domains.items():
            if not status['eligible']:
                continue
            _check_deadline(deadline)
            candidates = [r for r in available if old.family(r['dataset']) == domain]
            enriched = old.enrich_images(candidates, root, output)
            _check_deadline(deadline)
            byte_exclusions = {f'image_sha256:{value}' for value in old_images}
            selected, _ = old.group_closure(enriched, byte_exclusions)
            if not selected:
                status.update(eligible=False, status='blocked')
                status['blocking_reasons'].append({'code': 'no_rows_after_image_byte_exclusions'})
                continue
            for row in selected:
                row.update(metric=old.data.METRICS[domain], quality_metric=old.data.METRICS[domain],
                           evaluation_stage='independent_confirmation_v12')
            manifest = seal_manifest({'format': 'mlsys_adaptive_data_v3',
                'dataset': 'ASYMVLM_COMPRESSION_V12_'+domain,
                'metadata': {'study_sha256': study['sha256'], 'labels_loaded': False,
                    'fresh_identity_audit_passed': True, 'historical_sources_revalidated': True,
                    'complete_available_eligible_evaluation_split': True, 'history_sha256': history['sha256']},
                'splits': {'fit': [], 'screening': [], 'evaluation': selected, 'phi_train': []}})
            path = output/domain/'manifest.json'
            write_immutable(path, manifest)
            manifests[domain] = str(path)
            status.update(status='admitted', manifest_path=str(path), manifest_sha256=file_digest(path),
                          evaluation_rows=len(selected), image_exclusions_verified=True)
    except (TimeoutError, ValueError, FileNotFoundError, OSError) as exc:
        partial_error = f'{type(exc).__name__}: {exc}'
        domains = {name: {'dataset': name, 'eligible': False, 'status': 'blocked',
                          'blocking_reasons': [{'code': 'audit_incomplete', 'detail': partial_error}],
                          'labels_loaded': False} for name in old.FAMILIES}
        manifests = {}
    pins.update({str(path): file_digest(path) for path in (status_path, history_path, triage_path, Path(__file__))})
    result = old.seal({'format': FORMAT, 'study_sha256': study['sha256'],
        'status': 'partially_admitted' if manifests else 'blocked', 'domains': domains, 'manifests': manifests,
        'started_utc_epoch': started, 'deadline_utc_epoch': deadline, 'elapsed_seconds': time.time()-started,
        'labels_loaded': False, 'predictions_read': False, 'historical_artifacts_modified': False,
        'historical_artifacts': len(history['issues']), 'attempted_artifacts': len(receipts),
        'resolved_artifacts': sum(r['resolved'] for r in receipts),
        'remaining_unresolved_artifacts': len(history['issues'])-sum(r['resolved'] for r in receipts),
        'receipts': receipts, 'input_hashes': pins, 'source_files': source_files,
        'identity_inventory_sha256': digest(rows), 'frozen_exclusions_sha256': digest(exclusions),
        'scope_note': 'Complete eligible official evaluation rows only; no previously inspected data relabeled as fresh.',
        'audit_error': partial_error})
    write_immutable(output/'status.json', result)
    from .state import atomic_json
    atomic_json(Path(study['output_root'])/'full-suite/status.json', full_suite_readiness(study, result))
    print(json.dumps({'status': result['status'], 'attempted_artifacts': len(receipts),
        'resolved_artifacts': result['resolved_artifacts'], 'remaining_unresolved_artifacts': result['remaining_unresolved_artifacts'],
        'labels_loaded': False, 'path': str(output/'status.json')}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args(argv)
    publish(features.execution.checked(args.study))


if __name__ == '__main__':
    main()
