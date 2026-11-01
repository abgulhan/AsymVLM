"""Local ClusterB PBS transport repair for explicitly attached idle workers.

Self-SSH through clusterb-login-01 can fail even while local PBS and direct node
access work. On verified ClusterB hosts only, execute the exact frozen allocation
query script locally. Ownership, interactive/by-node state, lifetime, eight-GPU
idle checks, task/source verification, and exclusive node leases are unchanged.
This module neither submits jobs nor bypasses the normal idle-worker checks.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import socket

from . import compression_operations_v13 as frozen
from .compression_matrix_v8 import seal
from .followup_common import write_immutable
from .state import file_digest

FORMAT = 'mlsys_compression_operations_v13a'
_QUERY = frozen.clusterb.query_allocations


def allowed_host():
    host = socket.gethostname().split('.')[0]
    if host != 'clusterb-login-01' and not frozen.clusterb.HOST_PATTERN.fullmatch(host):
        raise ValueError('local PBS transport is restricted to verified ClusterB hosts')
    return host


def local_query(user, *, run=None):
    """Reuse the original generated query, including PBS-time normalization."""
    allowed_host()
    execute = frozen.clusterb.command if run is None else run

    def transport(argv):
        expected = ['ssh', *frozen.clusterb.SSH_OPTIONS, 'clusterb-login-01']
        if not isinstance(argv, list) or len(argv) != len(expected) + 1 or argv[:-1] != expected:
            raise ValueError('unexpected frozen allocation-query transport')
        command = shlex.split(argv[-1])
        if len(command) != 3 or command[:2] != ['python3', '-c']:
            raise ValueError('unexpected frozen allocation-query program')
        # No shell: quotes in the original script have already been decoded.
        return execute(command, timeout=30)

    return _QUERY(user, run=transport)


def receipt(config, config_path):
    if config['coordinator_role'] != 'clusterb':
        raise ValueError('local allocation query repair is ClusterB-only')
    allowed_host()
    paths = [Path(__file__), Path(frozen.__file__), Path(frozen.clusterb.__file__), Path(config_path)]
    value = seal({'format': FORMAT + '_receipt', 'configuration_sha256': config['sha256'],
        'study_sha256': config['study_sha256'], 'owner': config['user'],
        'input_hashes': {str(path.resolve()): file_digest(path) for path in paths},
        'change': 'run unchanged source-pinned ClusterB PBS allocation-query script locally, without self-SSH',
        'host_scope': ['clusterb-login-01', 'clusterb-gpu-[0-9]+'],
        'ownership_and_lifetime_validation_unchanged': True, 'gpu_idle_and_lease_validation_unchanged': True,
        'frozen_study_and_gpu_plans_changed': False, 'allocation_submission_authorized_by_this_wrapper': False})
    path = Path(config['state_root']) / 'local-pbs-transport-v13a.json'
    write_immutable(path, value)
    return value


def verify_receipt(value):
    for path, expected in value['input_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('local PBS repair source changed: ' + path)


@contextmanager
def bindings(value=None):
    original = frozen.clusterb.query_allocations

    def query(user):
        if value is not None:
            verify_receipt(value)
            if user != value['owner']:
                raise ValueError('local PBS repair requested for a different owner')
        return local_query(user)

    frozen.clusterb.query_allocations = query
    try:
        yield
    finally:
        frozen.clusterb.query_allocations = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inspect', 'owned-idle-worker'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-id')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seconds', type=int, default=3000)
    args = parser.parse_args(argv)
    config = frozen.read_config(args.config)
    if config['coordinator_role'] != 'clusterb':
        raise ValueError('local allocation query repair is ClusterB-only')
    allowed_host()
    if args.phase == 'inspect':
        payload = local_query(config['user'])
        result = frozen.clusterb.owned_allocations(payload, config['user'], frozen.time.time(),
            config['deadline_utc_epoch'])
        print(json.dumps({'format': FORMAT + '_inspection', 'host': allowed_host(),
            'owned_allocations': result, 'read_only': True}), flush=True)
        return
    if args.job_id is None or args.gpu is None:
        parser.error('--job-id and --gpu required for owned-idle-worker')
    if not 0 < args.seconds <= 10800:
        parser.error('--seconds must be between 1 and 10800')
    value = receipt(config, args.config)
    with bindings(value):
        result = frozen.clusterb_worker(config, args.job_id, args.gpu, args.seconds)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
