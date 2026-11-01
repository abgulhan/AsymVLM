"""Spare-capacity attachment using existing queue/claim/guard primitives.

No PBS submission, job cancellation, modifications to parent queues, or private
scheduler. This wrapper only admits the separate audit queue when the live main
coordinator reports no claimable work. Existing task locks/retries are reused.
"""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time

from . import adaptive_operations_v4 as clusterb
from .compression_operations_v13a import local_query
from .compression_operations_v14 import direct_idle_gpus
from . import overnight,sprint48
from .coverage_audit import ROOT,STUDY,read,prepare,report
from .followup_common import write_immutable
from .state import atomic_json,file_digest,digest

PREFIX='mlsys-coverage-audit-'


def sealed(value):return {**value,'sha256':digest(value)}


def setup(project):
    project=Path(project).resolve();root=project/ROOT
    audit=prepare(project)
    path=root/'operations/config.json'
    if path.exists():return sprint48.read_config(path)
    config=sealed({'format':sprint48.FORMAT,'cwd':str(project),'user':getpass.getuser(),
        'created_utc_epoch':audit['created_utc_epoch'],'deadline_utc_epoch':audit['physical_deadline_utc_epoch'],
        'maximum_submissions':1,'allocation_submission_disabled':True,'coordinator_role':'clusterb',
        'state_root':str(root/'operations/state'),
        'pbs_script':'scripts/mlsys_compression_v16_clustera.pbs',
        'plan_paths':list(audit['plans'].values()),'input_hashes':audit['input_hashes'],
        'audit_manifest':str(root/'audit.json'),'main_status':str(project/STUDY.parent/'coordinator/clusterb/status.json'),
        'stop_new_physical_utc_epoch':audit['stop_new_physical_utc_epoch'],'maximum_idle_gpus':2})
    write_immutable(path,config)
    return sprint48.read_config(path)


def main_has_priority(config, *, now=None):
    now=time.time() if now is None else now
    path=Path(config['main_status'])
    if not path.exists():return True,'main_status_missing'
    state=read(path)
    if not isinstance(state.get('time'),(int,float)) or not 0<=now-state['time']<=90:
        return True,'main_status_stale'
    if state.get('status') not in ('workers_active_or_waiting_for_idle_gpu','waiting_for_published_work_or_dependencies','complete','finished_with_failures'):
        return True,'main_state_uncertain'
    if state.get('ready_tasks',1)>0:return True,'main_has_ready_tasks'
    return False,'main_has_no_ready_tasks'


def idle_worker(config,job,gpu,seconds):
    host=socket.gethostname().split('.')[0]
    if (config['user']!=getpass.getuser() or type(seconds) is not int or not 600<=seconds<=1800
            or not clusterb.HOST_PATTERN.fullmatch(host) or type(gpu) is not int or gpu not in range(8)
            or not clusterb.JOB_PATTERN.fullmatch(job)):
        raise ValueError('explicit valid ClusterB job/node/GPU required')
    owned=clusterb.owned_allocations(local_query(config['user']),config['user'],time.time(),config['deadline_utc_epoch'])
    allocation=next((a for a in owned if a['job_id']==job and a['host']==host),None)
    if allocation is None:raise ValueError('not an owned live interactive allocation')
    root=Path(config['audit_manifest']).parent
    # This is the cross-workflow lease used by existing ClusterB idle workers.
    fd=os.open(f'/tmp/kv_day3_{host}_gpu{gpu}.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        info=os.fstat(fd)
        if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode):raise ValueError('invalid GPU lease')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=clusterb.command(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'])
        memory=clusterb.command(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'])
        if apps.strip() or memory.strip()!='0':raise ValueError('GPU is occupied')
        os.set_inheritable(fd,True)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),PBS_JOBID=job)
        end=time.monotonic()+min(seconds,allocation['seconds'],config['deadline_utc_epoch']-time.time())
        while end-time.monotonic()>=600 and time.time()<config['stop_new_physical_utc_epoch']:
            busy,reason=main_has_priority(config)
            if busy:
                print(json.dumps({'event':'audit_yield_to_main','reason':reason}),flush=True);break
            queues,_=sprint48.load_queues(config,'gpu')
            found=overnight.claim(queues,remaining_seconds=end-time.monotonic())
            if found is None:break
            task,plan,state_root,state,lock=found
            # Existing guard respects shared/exclusive node timing leases.
            guarded={**task,'argv':[str(Path(config['cwd'])/'.conda/asymvlm/bin/python'),'-B','-u','-m',
                'learned_score.mlsys_sprint.sprint48','guard','--mode','shared','--',*task['argv']]}
            overnight.run_claim((guarded,plan,state_root,state,lock),end,
                f'coverage-audit-{job}-{host}-{gpu}',lambda:False)
            report(root)
    finally:os.close(fd)


def once(config,config_path):
    if socket.gethostname().split('.')[0]!='clusterb-login-01':raise ValueError('attachment requires ClusterB login')
    busy,reason=main_has_priority(config)
    status={'time':time.time(),'status':'waiting_for_spare_capacity','reason':reason,'launched':[]}
    if busy or time.time()>=config['stop_new_physical_utc_epoch']:return status
    sessions=clusterb.tmux_sessions()
    available=max(0,2-sum(s.startswith(PREFIX) for s in sessions))
    if not available:return {**status,'status':'two_audit_workers_active'}
    queues,missing=sprint48.load_queues(config,'gpu')
    if sprint48.all_terminal(queues,missing):return {**status,'status':'audit_queue_terminal'}
    owned=clusterb.owned_allocations(local_query(config['user']),config['user'],time.time(),config['deadline_utc_epoch'])
    for allocation in sorted(owned,key=lambda a:a['host']):
        for gpu in direct_idle_gpus(allocation['host']):
            if available==0:return status
            name=PREFIX+allocation['host']+'-gpu'+str(gpu)
            if gpu_session_present(sessions,allocation['host'],gpu):continue
            remote=['bash',str(Path(config['cwd'])/'scripts/mlsys_coverage_audit_idle_worker.sh'),
                allocation['job_id'],str(gpu),str(min(1800,allocation['seconds'])),str(config_path)]
            log=Path(config['audit_manifest']).parent/'operations/logs'/(name+'-'+str(int(time.time()))+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            command=['ssh',*clusterb.SSH_OPTIONS,allocation['host'],shlex.join(remote)]
            clusterb.command(['tmux','new-session','-d','-s',name,
                'exec '+shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1'])
            sessions.add(name);available-=1
            status['launched'].append({'session':name,'job':allocation['job_id'],'gpu':gpu,'log':str(log)})
    return status


def gpu_session_present(sessions,host,gpu):
    """Do not attach during another worker's CPU-only model-loading interval."""
    return any(host in name and re.search(r'(?:^|[-_])gpu'+str(gpu)+r'(?:[-_]|$)',name) for name in sessions)


def watch(config,path):
    root=Path(config['audit_manifest']).parent
    with (root/'operations/attach.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.time()<config['deadline_utc_epoch']:
            try:result=once(config,path)
            except (ValueError,OSError,subprocess.SubprocessError) as exc:
                result={'time':time.time(),'status':'waiting_for_verified_spare_capacity','error':str(exc)}
            atomic_json(root/'operations/status.json',result)
            report(root)
            print(json.dumps(result),flush=True)
            if result['status']=='audit_queue_terminal':return result
            time.sleep(min(30,max(0,config['deadline_utc_epoch']-time.time())))
    return {'status':'deadline'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('setup','once','watch','idle-worker'))
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path)
    parser.add_argument('--job')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--seconds',type=int,default=1800)
    args=parser.parse_args(argv)
    path=args.config or args.project/ROOT/'operations/config.json'
    config=setup(args.project) if args.phase=='setup' else sprint48.read_config(path)
    if args.phase=='idle-worker':return idle_worker(config,args.job,args.gpu,args.seconds)
    if args.phase=='watch':result=watch(config,path)
    elif args.phase=='once':
        result=once(config,path)
        atomic_json(Path(config['audit_manifest']).parent/'operations/status.json',result)
    else:result={'status':'prepared','config':str(path)}
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
