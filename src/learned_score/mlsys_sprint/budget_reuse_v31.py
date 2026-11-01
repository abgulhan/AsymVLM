"""Provenance-checked reuse of physical Qwen labels, never interpolated labels."""
import json
from pathlib import Path

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import file_digest


def prepare(project, destination, model):
    destination = Path(destination)
    if destination.exists():
        return checked(destination)
    project = Path(project).resolve()
    base = project/'reproduction_results/mlsys_evidence_20261025_v18'
    index, provenance = {}, {}
    for stage in ('confirm_adapt_v25_20261027','confidence_v26_20261027',
                  'budget_head_v27_20261027', 'loss_frontier_v28_20261027'):
        for marker_path in sorted((base/stage/'physical').glob('*/batch*/rank0.complete.json')):
            marker = checked(marker_path)
            if not marker.get('passed'):
                continue
            metadata_path = marker_path.parent/'rank0.metadata.json'
            if not metadata_path.is_file():
                continue  # A repair-only shard without its native metadata is not reusable.
            metadata = json.loads(metadata_path.read_text())
            backbone = metadata['backbone']
            if (model['checkpoint_sha256'] not in metadata['checkpoints'].values()
                    or backbone['model'] != model['model_id']
                    or backbone.get('resolved_revision') != model['revision']
                    or backbone.get('resolution', {}).get('max_visual_patches') != model['max_visual_patches']
                    or backbone.get('position_policy') != 'original'
                    or backbone.get('question_only') is not True
                    or metadata.get('max_new_tokens') != 128 or metadata.get('generated_budget') != 512
                    or metadata.get('answer_inputs_to_scorer') is not False):
                continue
            # Reuse requires identical native runtime, model and scoring sources.
            for filename, sha in metadata['source_hashes'].items():
                if file_digest(filename) != sha:
                    raise ValueError('historical physical runtime changed: '+filename)
            provenance[str(marker_path)] = file_digest(marker_path)
            provenance[str(metadata_path)] = file_digest(metadata_path)
            pins = marker['input_hashes']
            for filename, sha in pins.items():
                path = Path(filename)
                if path.parent.name != 'rows':
                    continue
                if file_digest(path) != sha:
                    raise ValueError('historical physical row changed')
                row = json.loads(path.read_text())
                if (row.get('checkpoint_sha256') != model['checkpoint_sha256']
                        or row.get('selector') != 'scorer' or row.get('status') != 'ok'
                        or row.get('physical_measurement_performed') is not True
                        or row.get('max_new_tokens') != 128 or row.get('generated_budget') != 512):
                    continue
                feature_path = path.parent.parent/'features'/(row['request_id']+'.json')
                if pins.get(str(feature_path)) != file_digest(feature_path):
                    raise ValueError('unbound historical feature')
                feature = json.loads(feature_path.read_text())
                curve = feature['curves'][model['checkpoint_sha256']]
                key = row['request_id']+':'+str(row['selected_count'])
                candidate = dict(path=str(path), sha256=sha, feature=str(feature_path),
                    feature_sha256=pins[str(feature_path)], ranking_sha256=curve['ranking_sha256'],
                    marker=str(marker_path), metadata=str(metadata_path))
                if key in index:
                    other = json.loads(Path(index[key]['path']).read_text())
                    if (other['quality_score'],other['prediction']) != (row['quality_score'],row['prediction']):
                        raise ValueError('contradictory duplicate physical labels')
                else:
                    index[key] = candidate
    result = seal(dict(format='mlsys_physical_reuse_v31', rows=index, input_hashes=provenance,
        checkpoint_sha256=model['checkpoint_sha256'], counts=len(index)))
    write_immutable(destination, result)
    return result


def get(index, feature, count, execution_key):
    item = index['rows'].get(feature['request_id']+':'+str(count))
    if item is None:
        return None
    if file_digest(item['path']) != item['sha256'] or file_digest(item['feature']) != item['feature_sha256']:
        raise ValueError('reused physical artifact changed')
    row = json.loads(Path(item['path']).read_text())
    for k in ('request_id','image_sha256','input_visual_tokens','input_text_tokens','cache_contract'):
        if row[k] != feature[k]:
            raise ValueError('reused native input geometry/identity mismatch: '+k)
    if (row['checkpoint_sha256'] != feature['checkpoint_sha256']
            or item['ranking_sha256'] != feature['ranking_sha256']):
        raise ValueError('reused patch ranking differs')
    return dict(row, execution_key=execution_key, physical_reused=True, reuse_provenance=item,
        timing_isolated=False, timing_reused_not_an_independent_observation=True)
