"""Question-information ablation, isolated from all frozen V18--V23 code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from learned_score.mlsys_scorer.models import Rank32Scorer, require_finite, optimizer_step
from learned_score.mlsys_scorer.checkpoints import atomic_torch_save
from . import evidence_learning_v18 as learning
from . import attention_labels_v19 as labels
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_query_selector_v24'
ARMS = ('retrained', 'independent', 'contextual')


class QuestionScorer(Rank32Scorer):
    """Same rank-32 selector, with an optional frozen-LM question residual."""
    def __init__(self, hidden_size, mode, fixed_question=None):
        super().__init__(hidden_size)
        if mode not in ARMS:
            raise ValueError('unknown question scorer mode')
        self.mode = mode
        if mode == 'contextual':
            self.context = torch.nn.Linear(hidden_size, self.rank, bias=False)
            torch.nn.init.zeros_(self.context.weight)
        if mode == 'independent':
            if fixed_question is None or fixed_question.shape != (1, hidden_size):
                raise ValueError('one training-derived question vector required')
            require_finite(fixed_question, 'fixed question')
            if fixed_question.norm() == 0:
                raise ValueError('fixed question is zero')
            self.register_buffer('fixed_question', fixed_question.float().clone())
        self.context_encoder = None

    def forward(self, vision, question, question_context=None):
        if self.mode == 'independent':
            return super().forward(vision, self.fixed_question)
        if self.mode == 'retrained':
            return super().forward(vision, question)
        if question_context is None:
            if self.context_encoder is None:
                raise ValueError('contextual scorer requires answer-free question context')
            question_context = self.context_encoder(question)
        if question_context.shape != question.shape:
            raise ValueError('context and raw question spans differ')
        require_finite(question_context, 'question context')
        v = F.normalize(F.linear(F.normalize(vision.float(), dim=-1), self.visual.weight.float()), dim=-1)
        q = F.linear(F.normalize(question.float(), dim=-1), self.question.weight.float())
        q = F.normalize(q + F.linear(F.normalize(question_context.float(), dim=-1), self.context.weight.float()), dim=-1)
        return require_finite((v @ q.T).max(dim=-1).values, 'contextual patch scores')


@torch.no_grad()
def question_context(adapter, question):
    """Text-only decoder pass, no KV cache, image input, answer or model mutation."""
    if question.ndim != 2 or not len(question):
        raise ValueError('nonempty question tokens required')
    model = adapter.model
    embeddings = question.to(device=adapter.device, dtype=model.model.embed_tokens.weight.dtype)[None]
    positions = torch.arange(len(question), device=adapter.device)[None, None].expand(3, 1, -1)
    result = model.model(inputs_embeds=embeddings, attention_mask=torch.ones(
        (1, len(question)), dtype=torch.long, device=adapter.device),
        position_ids=positions, use_cache=False, output_attentions=False,
        output_hidden_states=False, return_dict=True)
    if getattr(result, 'past_key_values', None) is not None:
        raise ValueError('question-only context retained a cache')
    return require_finite(result.last_hidden_state[0].detach(), 'question-only hidden states')


def load(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('format') != FORMAT or payload['identity_sha256'] != digest(payload['identity']):
        raise ValueError('invalid question scorer checkpoint')
    marker = checked(Path(path).parent/'complete.json')
    if (marker['checkpoint_sha256'] != file_digest(path) or marker['identity_sha256'] != payload['identity_sha256']
            or marker['steps'] != payload['steps'] or payload['epoch'] != 3):
        raise ValueError('incomplete or changed question scorer')
    scorer = QuestionScorer(payload['hidden_size'], payload['mode'], payload['state_dict'].get('fixed_question')).to(device).float()
    scorer.load_state_dict(payload['state_dict'], strict=True)
    return scorer.eval().requires_grad_(False), payload


def validate_inputs(protocol):
    if protocol.get('experiment') != 'query_v24':
        raise ValueError('wrong campaign')
    for path, sha in protocol['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('changed question experiment input: '+path)


def extract(protocol, rank, world_size=8):
    from . import compression_execution_v16 as execution
    from .adaptive_data_v3 import load_manifest
    from .attention_pilot_v19 import attention_map
    from .evidence_annotation_v18 import qwen_patch_rectangles
    from learned_score.mlsys_scorer.backbone import prepared_inputs, frozen_versions, assert_frozen
    validate_inputs(protocol)
    if world_size != 8 or rank not in range(8):
        raise ValueError('eight deterministic shards required')
    root = Path(protocol['output_root']); marker = root/'labels'/f'rank{rank}.complete.json'
    if marker.exists():
        value = checked(marker)
        if value['protocol_sha256'] != protocol['sha256']:
            raise ValueError('extraction completion belongs to another protocol')
        for path, sha in value['input_hashes'].items():
            if file_digest(path) != sha: raise ValueError('extraction changed')
        return value
    manifest = load_manifest(protocol['manifest'])
    args = execution.runtime_args(learning.read(protocol['source_task']))
    args.manifest, args.split = Path(protocol['manifest']), 'scorer_train'
    adapter, dataset, _, reference = learning.load_pilot_runtime(args, manifest)
    adapter.dataset = dataset
    versions = frozen_versions(adapter.model)
    originals = {e['row_hash']: torch.load(protocol['original_labels'][e['row_hash']],
        map_location='cpu', weights_only=True) for e in manifest.splits['scorer_train'] if not e.get('paired_with')}
    original_records = list(originals.values())
    pins = {}
    for index, entry in enumerate(manifest.splits['scorer_train'][rank::world_size]):
        if time.time() >= protocol['physical_deadline_utc_epoch']:
            raise TimeoutError('question extraction deadline')
        path = root/'labels'/(entry['row_hash']+'.pt')
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['protocol_sha256'] != protocol['sha256'] or saved['row_hash'] != entry['row_hash']:
                raise ValueError('wrong resumed question features')
        else:
            old = originals[entry.get('paired_with', entry['row_hash'])]
            if old['image_sha256'] != entry['image_sha256']:
                raise ValueError('paired training image differs')
            row = dataset.row_for_entry(entry)
            prepared, visual, query = prepared_inputs(adapter, row, False)
            if len(prepared.answer_positions): raise ValueError('answer entered question features')
            rectangles = qwen_patch_rectangles(prepared).cpu()
            if not torch.equal(rectangles, old['patch_rectangles']) or visual.shape != old['vision'].shape:
                raise ValueError('paired native patch geometry differs')
            if not torch.allclose(visual.cpu().float(), old['vision'].float(), atol=.05, rtol=.02):
                raise ValueError('native visual features differ from audited training cache')
            if not entry.get('paired_with') and not torch.equal(query.cpu().bfloat16(), old['question']):
                raise ValueError('original training question embedding mismatch')
            context = question_context(adapter, query).cpu().bfloat16()
            if index == 0:
                from learned_score.mlsys_runtime.dynamic import generate_dynamic
                pruned = adapter.physically_prune(prepared, torch.arange(len(visual), device=adapter.device), 'original')
                before = generate_dynamic(adapter.model, pruned, max_new_tokens=8, slack=1, stop_on_eos=True)
                before_ids = before.ids.detach().cpu(); del before
                question_context(adapter, query)
                after = generate_dynamic(adapter.model, pruned, max_new_tokens=8, slack=1, stop_on_eos=True)
                if not torch.equal(before_ids, after.ids.detach().cpu()):
                    raise ValueError('question context changed subsequent full-token generation')
                del after, pruned
                canary = execution._canary(adapter, reference, prepared, dict(
                    input_visual_tokens=len(visual), input_text_tokens=prepared.inputs_embeds.shape[1]-len(visual)))
                write_immutable(root/'labels'/f'rank{rank}.canary.json', seal(dict(
                    protocol_sha256=protocol['sha256'], passed=True, native_parity=canary)))
            if entry.get('paired_with'):
                supervised, sv, sq = prepared_inputs(adapter, row, True)
                if not torch.equal(visual, sv) or not torch.equal(query, sq):
                    raise ValueError('teacher answer changed image/question features')
                raw = attention_map(adapter, supervised, supervised.answer_positions-1)
                del supervised, sv, sq
            else:
                raw = old['raw_attention']
            record = dict(raw_attention=raw, patch_rectangles=rectangles)
            target = labels.corrected_attention(record, labels.position_bias(original_records, entry['image_sha256']))
            saved = dict(format=FORMAT+'_training_features', protocol_sha256=protocol['sha256'],
                row_hash=entry['row_hash'], image_sha256=entry['image_sha256'], dataset=entry['dataset'],
                paired_with=entry.get('paired_with'), vision=old['vision'], question=query.cpu().bfloat16(),
                question_context=context, target=target.float(), answer_inputs=False,
                target_kind='position_corrected_teacher_attention_not_causal_necessity',
                original_label_sha256=file_digest(protocol['original_labels'][entry.get('paired_with',entry['row_hash'])]))
            atomic_torch_save(path, saved)
            del prepared, visual, query, context
        assert_frozen(adapter.model, versions)
        pins[str(path)] = file_digest(path)
        print(json.dumps(dict(event='query_features', rank=rank, completed=index+1)), flush=True)
    value = seal(dict(protocol_sha256=protocol['sha256'], rank=rank, input_hashes=pins, passed=True))
    write_immutable(marker, value)
    return value


def fit(protocol):
    validate_inputs(protocol); torch.set_num_threads(2)
    root = Path(protocol['output_root']); entries = learning.read(protocol['manifest'])['splits']['scorer_train']
    for rank in range(8):
        marker = checked(root/'labels'/f'rank{rank}.complete.json')
        if marker['protocol_sha256'] != protocol['sha256']: raise ValueError('wrong label completion')
        for p, sha in marker['input_hashes'].items():
            if file_digest(p) != sha: raise ValueError('training features changed')
    records = [torch.load(root/'labels'/(e['row_hash']+'.pt'), map_location='cpu', weights_only=True) for e in entries]
    multiplicities = Counter(r['image_sha256'] for r in records)
    fixed = torch.stack([r['question'].float().mean(0) for r in records]).mean(0, keepdim=True)
    initial, initial_payload = learning.load_scorer(protocol['checkpoint'])
    outputs = {}
    for mode in ARMS:
        torch.manual_seed(42)
        path = root/'training'/mode/'latest.pt'
        if (path.parent/'complete.json').exists(): load(path); outputs[mode]=str(path); continue
        scorer = QuestionScorer(initial.hidden_size, mode, fixed)
        missing, extra = scorer.load_state_dict(initial.state_dict(), strict=False)
        if extra or set(missing)-{'context.weight','fixed_question'}: raise ValueError('bad warm start')
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=.01)
        scorer.train().requires_grad_(True)
        identity = dict(protocol_sha256=protocol['sha256'], mode=mode,
            initial_checkpoint_sha256=file_digest(protocol['checkpoint']),
            reference_checkpoint_sha256=initial_payload['identity']['reference_checkpoint_sha256'],
            reference_identity=initial_payload['identity']['reference_identity'],
            training_ids=[r['row_hash'] for r in records], answer_inputs=False,
            image_balanced=True, epochs=3, seed=42, learning_rate=1e-4, weight_decay=.01)
        steps = 0
        for epoch in range(3):
            order = list(records); random.Random(42+epoch).shuffle(order)
            for record in order:
                if time.time() >= protocol['stop_new_work_utc_epoch']: raise TimeoutError('fit deadline')
                optimizer.zero_grad(set_to_none=True)
                scores = scorer(record['vision'], record['question'], record['question_context'])
                loss = learning.evidence_loss(scores, record['target']) / multiplicities[record['image_sha256']]
                optimizer_step(loss, scorer, optimizer, 1.); steps += 1
        payload = dict(format=FORMAT, hidden_size=scorer.hidden_size, mode=mode, identity=identity,
            identity_sha256=digest(identity), epoch=3, steps=steps,
            state_dict={k:v.detach().cpu() for k,v in scorer.state_dict().items()})
        atomic_torch_save(path, payload)
        write_immutable(path.parent/'complete.json', seal(dict(checkpoint_sha256=file_digest(path),
            identity_sha256=payload['identity_sha256'], steps=steps, passed=True)))
        outputs[mode]=str(path)
    write_immutable(root/'fit.json',seal(dict(protocol_sha256=protocol['sha256'],checkpoints=outputs,rows=len(records))))
    return outputs


def evaluate(task, rank, world_size):
    """Extend the native evaluator in this process only; do not edit old sources."""
    from . import evidence_evaluate_v18 as native
    from .adaptive_benchmark_v1 import select_exact_count
    from . import adaptive_experiment_v3 as prior
    original_runtime, original_load, original_selector = learning.load_pilot_runtime, native.load_candidate, native.make_selector
    state = {}
    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs); state['adapter'] = result[0]
        adapter=result[0];original_prepare=adapter.prepare_sample
        def prepare(row,include_answer=False):
            from .adaptive_data_v3 import dataset_family
            prepared=original_prepare(row,include_answer=include_answer)
            prepared.metadata['_query_v24_dataset']=dataset_family(row['_source_dataset'])
            return prepared
        adapter.prepare_sample=prepare
        return result
    def candidate(path, reference_sha, reference_payload, device):
        header = torch.load(path, map_location='cpu', weights_only=True)
        if header.get('format') != FORMAT:
            return original_load(path, reference_sha, reference_payload, device)
        scorer, payload = load(path, device)
        if (payload['identity']['reference_checkpoint_sha256'] != reference_sha or
                payload['identity']['reference_identity'] != reference_payload['identity']):
            raise ValueError('context scorer belongs to another backbone')
        scorer.context_encoder = lambda q: question_context(state['adapter'], q)
        return scorer, payload
    def selector(cell, scorer, n, p, count, observed, **kwargs):
        if task.get('query_control', 'correct') == 'correct':
            return original_selector(cell, scorer, n, p, count, observed, **kwargs)
        if cell['selector'] != 'scorer': raise ValueError('query substitution applies only to scorers')
        @torch.no_grad()
        def select(prepared):
            adapter = state['adapter']; question = prepared.metadata['question']
            donor = task['query_donors'].get(prepared.metadata['_query_v24_dataset']+'\0'+question)
            if donor is None or donor == question: raise ValueError('missing distinct query donor')
            ids = adapter.processor.tokenizer(donor, add_special_tokens=False).input_ids
            query = adapter.model.model.embed_tokens(torch.tensor(ids,device=adapter.device))
            visual = prepared.inputs_embeds[0, prepared.visual_positions]
            if (len(visual),prepared.inputs_embeds.shape[1]-len(visual)) != (n,p):
                raise ValueError('question substitution changed native geometry')
            indices = select_exact_count(scorer(visual,query),visual,count)
            observed.update(selected_patch_indices=indices.cpu().tolist(),input_visual_tokens=n,
                selected_patch_indices_sha256=digest(indices.cpu().tolist()), visual_retention=count/n,
                input_text_tokens=p,retained_visual_tokens=count,original_question_used_for_generation=True,
                scorer_question_control='different_same_dataset_length_matched_question')
            return indices
        return select, prior.Cell('scorer','uniform',0)
    learning.load_pilot_runtime, native.load_candidate, native.make_selector = runtime, candidate, selector
    try:
        return native.run(task, rank, world_size)
    finally:
        learning.load_pilot_runtime, native.load_candidate, native.make_selector = original_runtime, original_load, original_selector


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('extract','fit','evaluate'))
    parser.add_argument('--protocol',type=Path);parser.add_argument('--task',type=Path)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world-size',type=int,default=8)
    args=parser.parse_args(argv)
    result = evaluate(checked(args.task),args.rank,args.world_size) if args.phase=='evaluate' else (
        extract(checked(args.protocol),args.rank,args.world_size) if args.phase=='extract' else fit(checked(args.protocol)))
    print(json.dumps({'phase':args.phase,'complete':True}),flush=True)
    return result


if __name__=='__main__':main()
