"""Identity-selected public benchmark inventory for the frozen manuscript run.

Selection never uses outcomes. Actual scorer-training image/document identities
are excluded. Historical evaluation is disclosed, not mislabeled confirmation.
"""
import os
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .state import atomic_json, digest, file_digest
from .followup_common import write_immutable
from learned_score.mlsys_eval.common import seal_manifest

ROOT = Path('reproduction_results/mlsys_manuscript_v35_20261028')
TRAINING = (
    ('reproduction_results/mlsys_20261012/manifest.json', ('scorer_train',)),
    ('reproduction_results/mlsys_evidence_20261025_v18/data/manifest.json', ('scorer_train',)),
    ('reproduction_results/mlsys_evidence_20261025_v18/query_v24_20261027/data/manifest.json', ('scorer_train', 'fit')),
)
DOC_REPO = 'pixparse/docvqa-single-page-questions'
DOC_REV = '33136ef456fa5a3fe68568d6e31dda4eeff95b9b'


def family(name):
    text = name.lower()
    for key, value in (('infographic', 'InfographicVQA'), ('textvqa', 'TextVQA'),
                       ('docvqa', 'DocVQA'), ('chartqa', 'ChartQA'),
                       ('ocrvqa', 'OCRVQA_TESTCORE'), ('mme', 'MME')):
        if key in text:
            return value
    return name


def identity_keys(row):
    result = set()
    for key in ('image_sha256', 'group_id'):
        if row.get(key):
            result.add((key, str(row[key])))
    for key in ('image_id', 'document_id', 'question_id'):
        if row.get(key) is not None:
            result.add((family(row['dataset']), key, str(row[key])))
    return result


def exclusions(project):
    keys, receipts = set(), []
    for name, splits in TRAINING:
        path = project / name
        value = json.loads(path.read_text())
        selected = [r for s in splits for r in value['splits'].get(s, [])]
        if not selected:
            raise ValueError('no known actual training split in ' + name)
        for row in selected:
            keys.update(identity_keys(row))
        receipts.append(dict(path=str(path), sha256=file_digest(path),
                             splits=list(splits), excluded_training_rows=len(selected)))
    return keys, receipts


def acquire_doc():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    dest = Path('.data/native_sources/docvqa_33136ef4').resolve()
    def one(i):
        name = f'data/validation-{i:05d}-of-00004.parquet'
        existing = dest / name
        if existing.is_file():
            return str(existing)
        return hf_hub_download(DOC_REPO, name, repo_type='dataset', revision=DOC_REV,
                               token=False, local_dir=str(dest))
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(one, range(4)))
    atomic_json(ROOT/'data/doc-acquisition.json', dict(repository=DOC_REPO,
        revision=DOC_REV, files={p: file_digest(p) for p in files}, complete=True))
    print('DocVQA validation: all four shards available', flush=True)


def parquet_rows(project, paths, dataset, revision, metric):
    import pyarrow.parquet as pq
    rows = []
    for path in paths:
        sha = file_digest(path)
        names = pq.ParquetFile(path).schema_arrow.names
        columns = [n for n in ('question_id', 'image_id', 'other_metadata', 'image',
                                'image_url', 'question') if n in names]
        for index, raw in enumerate(pq.read_table(path, columns=columns).to_pylist()):
            image_sha = hashlib.sha256(raw['image']['bytes']).hexdigest()
            meta = raw.get('other_metadata') or {}
            image_id = str(raw.get('image_id') or meta.get('image') or raw.get('image_url') or image_sha)
            document = meta.get('ucsf_document_id')
            if dataset == 'DocVQA' and not document:
                raise ValueError('DocVQA document identity is absent')
            group = f'{dataset}:document:{document}' if document else f'{dataset}:image:{image_id}'
            row = dict(dataset=dataset, dataset_revision=revision, source_split='validation',
                question_id=str(raw['question_id']), image_id=image_id,
                image_sha256=image_sha, group_id=group, group_provenance='document_or_image_identity',
                source_format='parquet', source_file=str(path.resolve()),
                source_row_index=index, source_shard_sha256=sha, dataset_index=index,
                metric=metric, quality_metric=metric, question_sha256=digest(raw['question']))
            if document:
                row['document_id'] = str(document)
            row['row_hash'] = digest([dataset, row['question_id'], image_sha, row['question_sha256']])
            rows.append(row)
    return rows


def tsv_rows(project, dataset, metric):
    from .adaptive_data_v3 import _identity_tsv, _hash_tsv_images
    source = project/'.data/LMUData'/f'{dataset}.tsv'
    rows = _identity_tsv(source, dataset)
    _hash_tsv_images(rows, project)
    csv.field_size_limit(64*1024*1024)
    with source.open() as handle:
        questions = [r['question'] for r in csv.DictReader(handle, delimiter='\t')]
    for row, question in zip(rows, questions):
        row.update(metric=metric, quality_metric=metric, question_sha256=digest(question))
        row['row_hash'] = digest([family(dataset), row['question_id'], row['image_sha256'], row['question_sha256']])
    return rows


def ocr_core(project):
    """Materialize the available named TESTCORE only; never call it full OCRVQA."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    base = Path(os.environ.get('ASYMVLM_VLMEVAL_DATASETS', 'data/VLMEvalKit/DATASETS'))
    source = base/'OCRVQA_TESTCORE.tsv'
    dest = project/ROOT/'data/ocr-testcore.parquet'
    if not dest.exists():
        records = []
        with source.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                raw = Path(row['image_path'])
                candidates = (raw, base/'images/OCRVQA_TESTCORE'/raw,
                              base/'images/OCRVQA_TESTCORE'/f"{row['index']}.jpg",
                              base/'images/OCRVQA_TEST'/raw)
                found = next((p.resolve() for p in candidates if p.is_file()), None)
                if found is None:
                    raise FileNotFoundError('unavailable OCR TESTCORE image '+str(raw))
                answers = row['answer']
                try:
                    answers = json.loads(answers)
                except ValueError:
                    answers = [answers]
                if not isinstance(answers, list):
                    answers = [str(answers)]
                records.append(dict(question_id=str(row['index']), question=row['question'],
                    answers=[str(a) for a in answers], image=dict(bytes=found.read_bytes(), path=None)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(records), temp)
        temp.replace(dest)
    return parquet_rows(project, [dest], 'OCRVQA_TESTCORE', file_digest(source), 'exact')


def ordered(rows):
    """Balanced deterministic nested panels; MME always keeps both questions."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get('category', ''), row['group_id'])].append(row)
    buckets = defaultdict(list)
    for (category, group), members in groups.items():
        if family(members[0]['dataset']) == 'MME' and len(members) != 2:
            raise ValueError('incomplete MME pair after exclusions: '+group)
        buckets[category].append((digest(['v35-order', group]), members))
    for category in buckets:
        buckets[category].sort(key=lambda pair: pair[0])
    result = []
    for i in range(max(map(len, buckets.values()), default=0)):
        for category in sorted(buckets):
            if i < len(buckets[category]):
                result.extend(sorted(buckets[category][i][1], key=lambda r: r['row_hash']))
    return result


def prepare(project):
    project = Path(project).resolve()
    target = project/ROOT/'data/manifest.json'
    if target.exists():
        return json.loads(target.read_text())
    forbidden, training = exclusions(project)
    rows, coverage = [], {}
    specs = [('TextVQA', 'textvqa_9c0699cd/data', 3,
              '9c0699cd19768ac5ab97568f6b3cbac4c0062884', 'vqa'),
             ('DocVQA', 'docvqa_33136ef4/data', 4, DOC_REV, 'anls'),
             ('InfographicVQA', 'infographic_vqa_539088ef/normalized', 4,
              '539088ef8a8ada01ac8e2e6d4e372586748a265e', 'anls')]
    sources = {}
    for dataset, directory, expected, rev, metric in specs:
        paths = sorted((project/'.data/native_sources'/directory).glob('validation-*.parquet'))
        available = parquet_rows(project, paths, dataset, rev, metric)
        sources[dataset] = available
        coverage[dataset] = dict(source_rows=len(available), available_shards=len(paths),
            expected_shards=expected, public_split_complete=len(paths)==expected)
    for name, metric in [('ChartQA_TEST', 'relaxed'), ('MME', 'mme')]:
        sources[family(name)] = tsv_rows(project, name, metric)
        coverage[family(name)] = dict(source_rows=len(sources[family(name)]),
            public_split_complete=len(sources[family(name)])=={'ChartQA_TEST':2500, 'MME':2374}[name])
    try:
        sources['OCRVQA_TESTCORE'] = ocr_core(project)
        coverage['OCRVQA_TESTCORE'] = dict(source_rows=len(sources['OCRVQA_TESTCORE']),
            public_split_complete=False, evidence='TESTCORE subset, not full OCRVQA test')
    except (OSError, ValueError) as exc:
        coverage['OCRVQA_TESTCORE'] = dict(status='blocked_missing_data', error=str(exc), public_split_complete=False)
    for dataset, candidates in sources.items():
        excluded_groups = {r['group_id'] for r in candidates if identity_keys(r)&forbidden}
        eligible = [r for r in candidates if r['group_id'] not in excluded_groups
                    and not identity_keys(r)&forbidden]
        # Cluster byte-identical images even if their source IDs differ.
        aliases = {}
        for row in eligible:
            aliases.setdefault(row['image_sha256'], row['group_id'])
        if dataset != 'DocVQA':
            for row in eligible:
                row['group_id'] = aliases[row['image_sha256']]
        coverage[dataset].update(eligible_rows=len(eligible), excluded_rows=len(candidates)-len(eligible),
                                groups=len({r['group_id'] for r in eligible}))
        for order, row in enumerate(ordered(eligible)):
            row.update(panel_order=order, evaluation_stage='training_disjoint_public_rerun_not_untouched_confirmation')
            rows.append(row)
    if len({r['row_hash'] for r in rows}) != len(rows):
        raise ValueError('duplicate request identity')
    manifest = seal_manifest(dict(format='mlsys_adaptive_data_v3', dataset='MLSys manuscript public suite',
        splits={'evaluation': rows}, metadata=dict(coverage=coverage, actual_training_exclusions=training,
            selection_uses_outcomes=False, prior_evaluation_not_excluded=True,
            claim_is_training_disjoint_not_independent_confirmation=True)))
    write_immutable(target, manifest)
    atomic_json(project/ROOT/'data/coverage.json', coverage)
    print(json.dumps(coverage, sort_keys=True), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('acquire-doc', 'prepare'))
    args = parser.parse_args()
    acquire_doc() if args.phase == 'acquire-doc' else prepare(Path.cwd())


if __name__ == '__main__':
    main()
