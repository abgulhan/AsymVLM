"""Publish a separately authorized coverage-audit execution window.

The expired v2 campaign, scientific worker, panels, rules, and CPU measurements
are immutable inputs. Only execution identities, output locations, and the new
eight-hour window change. This module never launches workers or allocations.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import coverage_audit as original
from . import compression_execution_v16 as execution
from .followup_common import write_immutable
from .state import digest, file_digest


RUN_ID = 'coverage_audit_resume_20261025_164134'
PARENT = Path('reproduction_results/mlsys_compression_20261025_v16')
ROOT = PARENT / RUN_ID
SOURCE_ROOT = PARENT / 'coverage_audit_v2'
START = datetime(2026, 10, 25, 16, 41, 34, tzinfo=timezone.utc).timestamp()
WINDOW = {'created_utc_epoch': START, 'stop_new_physical_utc_epoch': START + 6.5 * 3600,
          'physical_deadline_utc_epoch': START + 7.5 * 3600,
          'deadline_utc_epoch': START + 8 * 3600}
FORMAT = 'mlsys_coverage_audit_resume_v1'


def checked(path):
    value = original.read(path)
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('sealed artifact changed: ' + str(path))
    return value


def _plain_output(path, root):
    """Never write through a symlink, nor outside the new campaign directory."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root):
        raise ValueError('resume output escapes its campaign')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('symlink in resume output path: ' + str(parent))
        if parent == root:
            break
    return path


def _copy_bytes(source, destination, root, expected):
    """Atomic no-overwrite byte-identical reuse, including on a repeated call."""
    destination = _plain_output(destination, root)
    if Path(source).is_symlink():
        raise ValueError('CPU reuse source must be a regular file')
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('CPU reuse input changed')
    if destination.exists():
        if not destination.is_file() or file_digest(destination) != expected:
            raise ValueError('existing reused CPU artifact differs')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)  # Atomic publication; cannot replace a user file.
        except FileExistsError:
            if destination.is_symlink() or file_digest(destination) != expected:
                raise ValueError('concurrent CPU artifact publication differs')
    finally:
        os.unlink(temporary)  # Only the exact temporary file created above.


def _publish(path, value, root):
    write_immutable(_plain_output(path, root), value)


def _relocate(value, source, target):
    if isinstance(value, str) and (value == str(source) or value.startswith(str(source) + '/')):
        return str(target / Path(value).relative_to(source))
    if isinstance(value, list):
        return [_relocate(v, source, target) for v in value]
    if isinstance(value, dict):
        return {k: _relocate(v, source, target) for k, v in value.items()}
    return value


def prepare(project, *, source_root=None, output_root=None):
    """Clone the predeclared unrun panel; never regenerate or select examples."""
    project = Path(project).resolve()
    source = Path(source_root or project / SOURCE_ROOT).absolute()
    target = Path(output_root or project / ROOT).absolute()
    parent = project / PARENT
    if (source.parent != parent or target.parent != parent or source == target
            or source.is_symlink() or target.name != RUN_ID):
        raise ValueError('separate direct campaign roots and the declared resume ID are required')
    _plain_output(target, parent)
    source_audit = source / 'audit.json'
    audit = checked(source_audit)
    if (audit.get('format') != original.FORMAT or audit.get('maximum_idle_gpus') != 2
            or audit.get('protect_main_run') is not True
            or any(audit.get(k) is not False for k in
                   ('independent_confirmation', 'scorer_training', 'candidate_promotion'))):
        raise ValueError('source is not the frozen additive development protocol')
    if any((source / 'physical').glob('*/rows/*.json')):
        raise ValueError('source has physical outcomes; outcome-blind unrun-panel resume required')
    if set(audit['plans']) != set(audit['tasks']):
        raise ValueError('source task/plan labels disagree')

    # Retain every old pin; the new authorization does not authorize changed code.
    pins = dict(audit['input_hashes'])
    old_tasks, old_plans = {}, {}
    for label in sorted(audit['tasks']):
        task_path, plan_path = Path(audit['tasks'][label]), Path(audit['plans'][label])
        if task_path != source / 'tasks' / (label + '.json') or plan_path != source / 'plans' / label / 'plan.json':
            raise ValueError('source task/plan reference escapes declared source root')
        task, plan = checked(task_path), checked(plan_path)
        execution.validate_task(task, 'physical', verify_files=False)
        if (task.get('audit_format') != original.FORMAT or task.get('audit_methods') != list(original.METHODS)
                or task.get('split') != 'screening' or task.get('audit_rule') != audit['rules'][task['model_tag']]
                or task.get('output_root') != str(source / 'physical' / label)):
            raise ValueError('source physical task differs from frozen scientific protocol')
        old_tasks[label], old_plans[label] = task, plan
        for source_pins in (task['input_hashes'], plan['input_hashes']):
            for filename, sha in source_pins.items():
                if filename in pins and pins[filename] != sha:
                    raise ValueError('conflicting source pins')
                pins[filename] = sha
        pins[str(task_path)], pins[str(plan_path)] = file_digest(task_path), file_digest(plan_path)
    pins[str(source_audit)] = file_digest(source_audit)
    for filename, sha in pins.items():
        if file_digest(filename) != sha:
            raise ValueError('frozen source input changed: ' + filename)

    cpu_sources = []
    for model in original.MODELS:
        paths = sorted((source / 'cpu' / model).glob('*.json'))
        if len(paths) != 8:
            raise ValueError('exactly eight frozen CPU probes per model required')
        checkpoint = next(t['model']['checkpoint'] for t in old_tasks.values() if t['model_tag'] == model)
        for path in paths:
            value = original.read(path)
            if (value.get('model') != model or value.get('request_id') != path.stem
                    or value.get('training_cache_not_evaluation') is not True
                    or value.get('answer_quality_measured') is not False
                    or value.get('source_sha256') != pins[str(Path(original.__file__).resolve())]
                    or value.get('scorer_checkpoint_sha256') != pins[str(Path(checkpoint).resolve())]):
                raise ValueError('CPU probe identity differs from frozen source')
            cpu_sources.append(path)
    cpu_sources.extend(source / 'cpu' / name for name in ('synthetic.json', 'certification.json'))
    cpu_pins = {str(path): file_digest(path) for path in cpu_sources}
    new_source = str(Path(__file__).resolve())
    worker_source = str(Path(__file__).with_name('coverage_audit_resume_worker.py').resolve())
    provenance = {str(source_audit): pins[str(source_audit)], new_source: file_digest(new_source),
                  worker_source: file_digest(worker_source)}
    copies = []
    for source_path in cpu_sources:
        destination = target / source_path.relative_to(source)
        _copy_bytes(source_path, destination, target, cpu_pins[str(source_path)])
        copies.append({'source': str(source_path), 'destination': str(destination),
                       'sha256': cpu_pins[str(source_path)], 'byte_identical': True})
    reuse = execution.sealed({'format': FORMAT + '_cpu_reuse', 'run_id': RUN_ID,
        'source_root': str(source), 'output_root': str(target), 'probes_per_model': 8,
        'probe_count': 24, 'cpu_inference_rerun': False, 'new_physical_evidence': False,
        'source_audit_sha256': audit['sha256'], 'input_hashes': {**provenance, **cpu_pins},
        'copies': copies})
    reuse_path = target / 'cpu-reuse.json'
    _publish(reuse_path, reuse, target)
    provenance[str(reuse_path)] = file_digest(reuse_path)

    tasks, plans = {}, {}
    for label in sorted(old_tasks):
        old_task, old_plan = old_tasks[label], old_plans[label]
        task = {k: deepcopy(v) for k, v in old_task.items() if k != 'sha256'}
        task.update(output_root=str(target / 'physical' / label),
                    deadline_utc_epoch=WINDOW['physical_deadline_utc_epoch'], resume_run_id=RUN_ID)
        task['input_hashes'].update(provenance)
        task_path = target / 'tasks' / (label + '.json')
        _publish(task_path, execution.sealed(task), target)
        records = _relocate(deepcopy(old_plan['tasks']), source, target)
        for record in records:
            if record.get('depends_on'):
                raise ValueError('unexpected dependencies in frozen source shards')
            record['id'] += '-resume-20261025-164134'
            module_index = record['argv'].index('-m') + 1
            if record['argv'][module_index] != 'learned_score.mlsys_sprint.coverage_audit':
                raise ValueError('unexpected physical worker in source plan')
            record['argv'][module_index] = 'learned_score.mlsys_sprint.coverage_audit_resume_worker'
        plan = {k: deepcopy(v) for k, v in old_plan.items() if k != 'sha256'}
        plan.update(tasks=records, created_utc_epoch=START,
                    deadline_utc_epoch=WINDOW['physical_deadline_utc_epoch'], resume_run_id=RUN_ID)
        # Original task/plan remain provenance inputs; the new task is additionally pinned.
        plan['input_hashes'].update(provenance)
        plan['input_hashes'][str(task_path)] = file_digest(task_path)
        plan_path = target / 'plans' / label / 'plan.json'
        _publish(plan_path, execution.sealed(plan), target)
        tasks[label], plans[label] = str(task_path), str(plan_path)

    resumed = {k: deepcopy(v) for k, v in audit.items() if k != 'sha256'}
    resumed.update(**WINDOW, tasks=tasks, plans=plans, run_id=RUN_ID,
        output_root=str(target), source_audit=str(source_audit), source_audit_sha256=audit['sha256'],
        execution_window_is_new_user_authorization=True, scientific_protocol_unchanged=True,
        source_physical_rows=0, cpu_reuse_receipt=str(reuse_path),
        operational_correctness_fix='synchronized whole coverage-greedy selector 60-second cap; '
                                  'overrun recorded as timeout before inference, never fallback')
    resumed['input_hashes'].update(provenance)
    resumed['input_hashes'].update(cpu_pins)
    # Verify source/copy stability before publishing the runnable audit manifest.
    for filename, sha in {**pins, **cpu_pins, **provenance}.items():
        if file_digest(filename) != sha:
            raise ValueError('source changed during resume publication: ' + filename)
    for copy in copies:
        if file_digest(copy['destination']) != copy['sha256']:
            raise ValueError('CPU reuse changed during publication')
    resumed = execution.sealed(resumed)
    _publish(target / 'audit.json', resumed, target)
    outputs = [target / 'audit.json', reuse_path, *map(Path, tasks.values()), *map(Path, plans.values())]
    receipt = execution.sealed({'format': FORMAT + '_receipt', 'run_id': RUN_ID, **WINDOW,
        'source_audit_sha256': audit['sha256'], 'audit_sha256': resumed['sha256'],
        'scientific_protocol_unchanged': resumed['panels'] == audit['panels'] and resumed['rules'] == audit['rules'],
        'panels_regenerated': False, 'physical_outcomes_read': False,
        'physical_workers_launched': False, 'cpu_probe_count_reused': 24,
        'operational_correctness_fix': resumed['operational_correctness_fix'],
        'physical_shards': sum(len(p['tasks']) for p in old_plans.values()),
        'input_hashes': {**pins, **cpu_pins, new_source: provenance[new_source],
                         worker_source: provenance[worker_source]},
        'output_hashes': {str(path): file_digest(path) for path in outputs}})
    _publish(target / 'resume-receipt.json', receipt, target)
    return resumed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare',))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = prepare(args.project)
    print(json.dumps({'run_id': RUN_ID, 'audit_sha256': result['sha256'],
                      'output_root': result['output_root'], 'plans': result['plans'], **WINDOW}))


if __name__ == '__main__':
    main()
