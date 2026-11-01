#!/usr/bin/env python
"""Copy result-summary JSONs while removing machine-specific fields.

Drops host/job/user fields and rewrites absolute paths to repository-relative ones.
Usage: python evaluation/tools/sanitize_results.py SRC DST [--strip-prefix PREFIX ...]
"""
import argparse
import json
import re
from pathlib import Path

DROP = {'host', 'hostname', 'hosts', 'job', 'job_id', 'jobs', 'allocation_job_id', 'user', 'owner', 'mail_to',
        'node', 'nodes', 'traceback', 'worker_id', 'cluster', 'account', 'queue', 'pid', 'tag'}
HOSTLIKE = re.compile(r'\b(x\d{4}c\d+s\d+b\d+n\d+|[a-z]+-gpu-\d+[\w.-]*|[a-z]+-login-\d+[\w.-]*|\d+\.[a-z]+-[a-z]{2}s-[\w.-]+)\b', re.I)


def clean_str(s, prefixes):
    for p in prefixes:
        s = s.replace(p, '')
    s = re.sub(r'(?:/[\w.@+-]+)+/hub/(models--)', r'$HF_HOME/hub/\1', s)
    s = re.sub(r'/(?:[\w.@+-]+/)*?(reproduction_results|manuscript|scripts|\.data)/', r'\1/', s)
    return HOSTLIKE.sub('<host>', s)


def clean(o, prefixes):
    if isinstance(o, dict):
        return {clean_str(k, prefixes): clean(v, prefixes) for k, v in o.items() if k.lower() not in DROP}
    if isinstance(o, list):
        return [clean(x, prefixes) for x in o]
    if isinstance(o, str):
        return clean_str(o, prefixes)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('src', type=Path); ap.add_argument('dst', type=Path)
    ap.add_argument('--strip-prefix', nargs='*', default=[])
    a = ap.parse_args()
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    a.dst.write_text(json.dumps(clean(json.loads(a.src.read_text()), sorted(a.strip_prefix, key=len, reverse=True)), indent=1))


if __name__ == '__main__':
    main()
