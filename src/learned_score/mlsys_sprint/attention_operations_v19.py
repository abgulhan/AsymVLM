"""Reuse the existing scheduler/leases with a separate attention-pilot namespace."""
import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from . import evidence_operations_v18 as base
from . import evidence_idle_clusterb_v18 as idle
from .attention_pilot_v19 import ROOT,source_pins
from .compression_matrix_v8 import checked
from .evidence_catalog_v18b import cached_catalog
from .state import file_digest


def validate_protocol(path):
    value=checked(path)
    root=Path(value['cwd']).resolve()/ROOT
    start=value['created_utc_epoch']
    if (Path(path).resolve()!=root/'protocol.json' or Path(value['output_root']).resolve()!=root
            or value.get('experiment')!='attention_v19' or value.get('maximum_gpu_workers')!=8
            or value['stop_new_work_utc_epoch']!=start+10800
            or value['physical_deadline_utc_epoch']!=start+12600
            or value['deadline_utc_epoch']!=start+14400):
        raise ValueError('attention pilot identity or four-hour deadlines changed')
    return value


def configure():
    original_sources=base.source_paths
    def sources(project):
        return list(dict.fromkeys(original_sources(project)+[Path(__file__).resolve(),
            *[Path(p) for p in source_pins()],Path(project)/'scripts/mlsys_attention_v19_idle_clusterb.sh']))
    base.ROOT=ROOT
    base.JOB_NAME='mlsys-attention-v19'
    base.PBS_SCRIPT='scripts/mlsys_attention_v19_clustera.pbs'
    base.WORKER_SCRIPT='scripts/mlsys_attention_v19_worker.sh'
    base.source_paths=sources
    base.validate_protocol=validate_protocol


def clusterb_worker(config,gpu,seconds):
    host=socket.gethostname().split('.')[0]
    if host!='clusterb-gpu-01' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or type(gpu) is not int or not 0<=gpu<8:
        raise ValueError('explicit idle ClusterB GPU required')
    # This existing allocation is revalidated, never submitted, altered or terminated.
    payload=json.loads(subprocess.check_output(['qstat','-f','-F','json',idle.JOB],text=True,timeout=30))
    allocation=idle.validate_allocation(config,payload,host,gpu,time.time())
    stop=[False]
    previous={sig:signal.signal(sig,lambda *_:stop.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with base.allocation_checks.gpu_lease(host,gpu) as fd:
            return base.run_claims(config,allocation,seconds,fd,lambda:stop[0])
    except BlockingIOError:
        return dict(status='gpu_lease_occupied',claims_run=0)
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def main(argv=None):
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('setup','watch','inspect','worker','clusterb-worker'))
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--config',type=Path,default=ROOT/'operations/config.json')
    parser.add_argument('--gpu',type=int);parser.add_argument('--seconds',type=int,default=3300)
    args=parser.parse_args(argv)
    with cached_catalog():
        if args.phase=='setup': result={'status':'prepared','config_sha256':base.build_config(args.protocol)['sha256']}
        elif args.phase=='worker': result=base.worker(base.read_config(args.config),args.seconds)
        elif args.phase=='clusterb-worker': result=clusterb_worker(base.read_config(args.config),args.gpu,args.seconds)
        else: result=base.watch(args.config,once=args.phase=='inspect',dry_run=args.phase=='inspect')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
