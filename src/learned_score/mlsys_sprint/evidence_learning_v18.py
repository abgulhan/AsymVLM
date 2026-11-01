"""Isolated, answer-free warm-start fits for the v18 evidence supervision pilot.

Only the small rank-32 scorer is loaded. Both arms use exactly the same accepted
examples and update schedule. Evidence distributions are already smoothed in
the audited annotation cache; control targets retain the old saliency loss.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from .state import digest, file_digest
from .followup_common import write_immutable
from .adaptive_data_v3 import ManifestDatasetV3, dataset_family

FORMAT = 'mlsys_evidence_learning_v18'
TASK_FORMAT = FORMAT + '_task'
CHECKPOINT_FORMAT = FORMAT + '_checkpoint'
NAMESPACE = 'reproduction_results/mlsys_evidence_20261025_v18'
RECIPE = dict(epochs=3, seed=42, learning_rate=1e-4, weight_decay=.01,
              max_grad_norm=1., evidence_smoothing=.05)


def quality_metric(dataset):
    family=dataset_family(dataset)
    if family not in ('DocVQA','TextVQA'):
        raise ValueError('v18 pilot supports only DocVQA/TextVQA')
    return 'anls' if family=='DocVQA' else 'vqa'


class EvidenceDataset(ManifestDatasetV3):
    """Preserve immutable source rows; decorate only the runtime parquet row."""
    def read_entry(self,entry):
        if entry not in self.allowed.get(entry['row_hash'],[]):
            raise ValueError('entry outside explicitly permitted pilot split')
        if entry.get('source_format')=='tsv':
            raise ValueError('pilot permits only the pinned official parquet sources')
        row=self.native.row_for_entry(self.native.allowed[entry['row_hash']])
        return {**row,'_manifest_row_hash':entry['row_hash'],'_source_dataset':entry['dataset'],
                'group_id':entry['group_id'],'metric':quality_metric(entry['dataset']),
                'category':entry.get('category','')}
    row_for_entry=read_entry


def load_pilot_runtime(args,manifest):
    """Inject only the dataset adapter while preserving the established model."""
    from . import adaptive_data_v3 as data
    from . import adaptive_experiment_v3 as prior
    original=data.ManifestDatasetV3
    if original is not ManifestDatasetV3:
        raise RuntimeError('pilot runtime adapter cannot nest another dataset adapter')
    data.ManifestDatasetV3=EvidenceDataset
    try:
        return prior.load_runtime(args,manifest)
    finally:
        data.ManifestDatasetV3=original


def read(path):
    return json.loads(Path(path).read_text())


def sealed(value):
    return {**value, 'sha256': digest(value)}


def validate_task(task, *, verify_files=True):
    if (task.get('format') != TASK_FORMAT or
            task.get('sha256') != digest({k:v for k,v in task.items() if k != 'sha256'})):
        raise ValueError('invalid v18 learning task seal')
    if task.get('arm') not in ('evidence', 'control'):
        raise ValueError('training arm must be evidence or control')
    expected = Path(task['cwd']).resolve()/NAMESPACE
    output = Path(task['output_root']).resolve()
    if output == expected or not output.is_relative_to(expected):
        raise ValueError('v18 fit must use its own output subdirectory')
    deadline = task.get('deadline_utc_epoch')
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ValueError('finite training deadline required')
    if task.get('recipe', RECIPE) != RECIPE:
        raise ValueError('approved fixed training recipe changed')
    if verify_files:
        required = [task['reference_checkpoint'], task['cache_index'], str(Path(__file__).resolve())]
        for path in required:
            if task.get('input_hashes', {}).get(str(Path(path).resolve())) != file_digest(path):
                raise ValueError('missing or changed required fit pin: '+str(path))
        for path, sha in task['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('changed fit input: '+path)
    return task


def validate_cache(record, entry, reference_sha=None):
    import torch
    from learned_score.mlsys_scorer.models import require_finite
    if record.get('format') != 'mlsys_evidence_cache_v18' or record.get('row_hash') != entry['row_hash']:
        raise ValueError('cache format/request identity mismatch')
    forbidden = {'answer', 'answers', 'answer_ids', 'answer_tokens', 'answer_positions', 'input_ids'}
    if forbidden & record.keys() or record.get('answer_free_embeddings') is not True:
        raise ValueError('cache must contain audited answer-free scorer inputs')
    if record.get('features_inputs_include_answer', False):
        raise ValueError('answers cannot enter scorer features')
    if reference_sha and record.get('target_model_checkpoint_sha256') != reference_sha:
        raise ValueError('cache reference scorer differs from warm start')
    vision, question, saliency = (record[k] for k in ('vision','question','saliency'))
    if not all(isinstance(tensor,torch.Tensor) for tensor in (vision,question,saliency)):
        raise ValueError('cache embedding values and saliency must be tensors')
    if (vision.ndim != 2 or question.ndim != 2 or not all(vision.shape) or
            not all(question.shape) or vision.shape[1] != question.shape[1] or
            saliency.shape != (len(vision),)):
        raise ValueError('cache embedding/saliency geometry mismatch')
    for name, tensor in (('vision',vision),('question',question),('saliency',saliency)):
        if not isinstance(tensor,torch.Tensor):
            raise ValueError('cache embedding values must be tensors')
        require_finite(tensor,name)
    target = record.get('evidence_targets')
    if record.get('annotation_status') != 'ok' or target is None:
        raise ValueError('both arms require the same usable annotation examples')
    if (record.get('evidence_smoothing') != .05 or target.shape != saliency.shape or
            not torch.isfinite(target).all() or abs(float(target.sum())-1.) > 1e-5 or
            bool((target < .05/len(target)-1e-7).any())):
        raise ValueError('evidence target must be normalized and smoothed exactly once')
    return record


def training_entries(index, *, verify_files=True):
    if index.get('format') != 'mlsys_evidence_cache_index_v18' or index.get('completed') is not True:
        raise ValueError('complete evidence cache index required')
    entries, seen = [], set()
    for entry in index['entries']:
        if entry['row_hash'] in seen:
            raise ValueError('duplicate training cache identity')
        seen.add(entry['row_hash'])
        if entry.get('annotation_status') != 'ok':
            continue
        if entry.get('evidence_smoothing') != .05:
            raise ValueError('cache index smoothing differs')
        if verify_files and file_digest(entry['path']) != entry['sha256']:
            raise ValueError('changed training cache record')
        if verify_files and entry.get('annotation_path') and file_digest(entry['annotation_path']) != entry['annotation_sha256']:
            raise ValueError('changed source annotation')
        entries.append(entry)
    if not entries:
        raise ValueError('no valid annotations; do not train empty/no-evidence targets')
    return sorted(entries, key=lambda e:(e['dataset'],e['row_hash']))


def evidence_loss(scores, target):
    """Cross entropy against normalized teacher targets, never softmax twice."""
    import torch
    from learned_score.mlsys_scorer.models import require_finite
    scores, target = require_finite(scores.float(),'scores'), require_finite(target.float(),'target')
    if (scores.ndim != 1 or scores.shape != target.shape or not len(scores) or
            bool((target < 0).any()) or not torch.isclose(target.sum(),target.new_tensor(1.),atol=1e-5)):
        raise ValueError('normalized equal-length distribution required')
    standardized = (scores-scores.mean())/scores.std(unbiased=False).clamp_min(1e-8)
    return -(target.detach()*standardized.log_softmax(0)).sum()


def save_checkpoint(path, scorer, optimizer, identity, epoch, next_index, steps):
    from learned_score.mlsys_scorer.checkpoints import atomic_torch_save
    payload = dict(format=CHECKPOINT_FORMAT, identity=identity, identity_sha256=digest(identity),
        variant='rank32', hidden_size=scorer.hidden_size,
        state_dict={k:v.detach().cpu() for k,v in scorer.state_dict().items()},
        optimizer=optimizer.state_dict(), epoch=epoch,next_index=next_index,steps=steps)
    atomic_torch_save(path,payload)


def load_scorer(path, device='cpu', *, require_complete=True):
    import torch
    from learned_score.mlsys_scorer.models import build_scorer, require_finite
    payload = torch.load(path,map_location=device,weights_only=True)
    if (payload.get('format') != CHECKPOINT_FORMAT or payload.get('variant') != 'rank32' or
            payload.get('identity_sha256') != digest(payload['identity'])):
        raise ValueError('invalid isolated v18 scorer checkpoint')
    if require_complete:
        marker = read(Path(path).parent/'complete.json')
        if (marker.get('checkpoint_sha256') != file_digest(path) or
                marker.get('identity_sha256') != payload['identity_sha256'] or
                marker.get('steps') != payload['steps'] or payload['epoch'] != 3 or payload['next_index'] != 0):
            raise ValueError('v18 checkpoint is incomplete or changed')
    scorer = build_scorer('rank32',payload['hidden_size']).to(device).float()
    scorer.load_state_dict(payload['state_dict'],strict=True)
    for parameter in scorer.parameters():
        require_finite(parameter,'checkpoint parameter')
    return scorer.eval(),payload


def run(task, device='cpu'):
    import torch
    from learned_score.mlsys_scorer.checkpoints import load_scorer as load_original
    from learned_score.mlsys_scorer.models import saliency_loss, optimizer_step
    from .benchmark import require_completed_checkpoint
    validate_task(task)
    entries = training_entries(read(task['cache_index']))
    torch.manual_seed(42)
    scorer, original = load_original(task['reference_checkpoint'], device)
    require_completed_checkpoint(task['reference_checkpoint'],original)
    if original['variant'] != 'rank32':
        raise ValueError('pilot warm start must be existing rank32 scorer')
    reference_sha = file_digest(task['reference_checkpoint'])
    identity = dict(format=FORMAT, arm=task['arm'], recipe=RECIPE,
        task_sha256=task['sha256'], cache_index_sha256=file_digest(task['cache_index']),
        reference_checkpoint_sha256=reference_sha, reference_identity=original['identity'],
        source_sha256=file_digest(__file__), rows=[e['row_hash'] for e in entries],
        cache_hashes={e['row_hash']:e['sha256'] for e in entries},
        dataset_counts=dict(Counter(e['dataset'] for e in entries)), answer_inputs=False)
    output = Path(task['output_root'])
    output.mkdir(parents=True,exist_ok=True)
    write_immutable(output/'identity.json',identity)
    latest, marker = output/'latest.pt',output/'complete.json'
    if marker.exists():
        _, payload = load_scorer(latest,require_complete=True)
        if payload['identity'] != identity:
            raise ValueError('completed fit identity changed')
        return read(marker)
    scorer.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=.01)
    start_epoch,next_index,steps = 0,0,0
    if latest.exists():
        _, prior = load_scorer(latest,device,require_complete=False)
        if prior['identity'] != identity:
            raise ValueError('resume training identity differs')
        scorer.load_state_dict(prior['state_dict'],strict=True)
        optimizer.load_state_dict(prior['optimizer'])
        start_epoch,next_index,steps = prior['epoch'],prior['next_index'],prior['steps']
    else:
        save_checkpoint(latest,scorer,optimizer,identity,0,0,0)
    history = []
    for epoch in range(start_epoch,3):
        order = list(entries)
        random.Random(42+epoch).shuffle(order)
        for index in range(next_index if epoch == start_epoch else 0,len(order)):
            if time.time() >= task['deadline_utc_epoch']:
                raise TimeoutError('fit deadline reached; last atomic checkpoint retained')
            entry = order[index]
            if file_digest(entry['path']) != entry['sha256']:
                raise ValueError('cache changed during fit')
            record = validate_cache(torch.load(entry['path'],map_location=device,weights_only=True),entry,reference_sha)
            if record['vision'].shape[1] != scorer.hidden_size:
                raise ValueError('warm start and feature dimension differ')
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(record['vision'].detach(),record['question'].detach())
            loss = (evidence_loss(scores,record['evidence_targets']) if task['arm']=='evidence'
                    else saliency_loss(scores,record['saliency']))
            optimizer_step(loss,scorer,optimizer,1.)
            steps += 1
            history.append(float(loss.detach()))
            save_checkpoint(latest,scorer,optimizer,identity,epoch,index+1,steps)
        save_checkpoint(latest,scorer,optimizer,identity,epoch+1,0,steps)
        print(json.dumps(dict(event='v18_fit_epoch',arm=task['arm'],epoch=epoch+1,steps=steps)),flush=True)
    result = dict(format=FORMAT+'_complete',identity_sha256=digest(identity),checkpoint_sha256=file_digest(latest),
        steps=steps,rows=len(entries),dataset_counts=identity['dataset_counts'],arm=task['arm'],
        recent_loss_mean=sum(history)/len(history) if history else None,passed=True)
    write_immutable(marker,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    print(json.dumps(run(read(args.task),args.device)),flush=True)


if __name__=='__main__':
    main()
