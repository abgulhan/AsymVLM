"""Catalog-only SHA memoization for the unchanged v18 idle-ClusterB workers.

The catalog repeatedly verifies two plans containing the same 9.47GB teacher
checkpoint while holding its global lock. This additive launcher caches only
those catalog checks, per process and only while filesystem identity is stable.
Fresh pre/post-task checks in worker.py, overnight.py, and evidence_operations
remain untouched. No source, plan, allocation, deadline, or lease is replaced.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import stat

from . import evidence_idle_clusterb_v18 as idle
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import file_digest

SCRIPT = 'scripts/mlsys_evidence_v18b_idle_clusterb.sh'


def identity(path):
    resolved = Path(path).resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('catalog source must be a regular file')
    return (str(resolved), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


class CatalogHashes:
    def __init__(self):
        self.values = {}

    def digest(self, path):
        before = identity(path)
        previous = self.values.get(before[0])
        if previous is not None and previous[0] == before:
            return previous[1]
        actual = file_digest(before[0])
        after = identity(path)
        if before != after:
            self.values.pop(before[0], None)
            raise ValueError('catalog source changed while hashing')
        self.values[before[0]] = (before, actual)
        return actual

    def verify(self, plan):
        for path, expected in plan.get('input_hashes', {}).items():
            if self.digest(Path(plan['cwd']) / path) != expected:
                raise ValueError(f'experiment input changed: {path}')


@contextmanager
def cached_catalog():
    """Replace only sprint48's catalog verifier, only during load_queues."""
    base = idle.base
    original_load = base.load_queues
    hashes = CatalogHashes()

    def load(config):
        original_verify = base.sprint48.verify_sources
        base.sprint48.verify_sources = hashes.verify
        try:
            return original_load(config)
        finally:
            base.sprint48.verify_sources = original_verify

    base.load_queues = load
    try:
        yield hashes
    finally:
        base.load_queues = original_load


def prepare(parent_path):
    parent = checked(parent_path)
    config = idle.base.read_config(parent['parent_config'])
    if (parent.get('format') != 'mlsys_evidence_v18_idle_clusterb'
            or parent.get('parent_sha256') != config['sha256']
            or parent.get('source_sha256') != file_digest(idle.__file__)
            or parent.get('launcher_sha256') != file_digest(Path(config['cwd'])/idle.SCRIPT)
            or parent.get('transition_sha256') != file_digest(parent['transition'])):
        raise ValueError('original idle-ClusterB configuration or pinned sources changed')
    value = seal(dict(format='mlsys_evidence_v18b_catalog_cache',
        parent_config=str(Path(parent_path).resolve()), parent_file_sha256=file_digest(parent_path),
        parent_sha256=parent['sha256'], source_sha256=file_digest(__file__),
        launcher_sha256=file_digest(Path(config['cwd'])/SCRIPT),
        memoization_scope='catalog_only_process_local', fresh_task_checks_preserved=True))
    path = Path(config['study_root'])/'operations/clusterb-config-v18b.json'
    write_immutable(path, value)
    return value


def worker(path, gpu, seconds):
    launch = checked(path)
    if (launch.get('format') != 'mlsys_evidence_v18b_catalog_cache'
            or launch.get('source_sha256') != file_digest(__file__)
            or launch.get('parent_file_sha256') != file_digest(launch['parent_config'])
            or launch.get('memoization_scope') != 'catalog_only_process_local'
            or launch.get('fresh_task_checks_preserved') is not True):
        raise ValueError('catalog-cache launcher identity changed')
    parent = checked(launch['parent_config'])
    config = idle.base.read_config(parent['parent_config'])
    if (parent['sha256'] != launch['parent_sha256']
            or launch['launcher_sha256'] != file_digest(Path(config['cwd'])/SCRIPT)):
        raise ValueError('original idle worker identity or additive launcher changed')
    with cached_catalog():
        return idle.worker(launch['parent_config'], gpu, seconds)


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'worker'])
    parser.add_argument('--parent-config', type=Path, default=idle.base.ROOT/'operations/clusterb-config.json')
    parser.add_argument('--config', type=Path, default=idle.base.ROOT/'operations/clusterb-config-v18b.json')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seconds', type=int, default=12000)
    args = parser.parse_args(argv)
    result = prepare(args.parent_config) if args.phase == 'prepare' else worker(args.config,args.gpu,args.seconds)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
