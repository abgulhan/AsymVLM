#!/usr/bin/env python
"""Rewrite ``source_file`` / ``image_path`` entries of a manifest to paths under ASYMVLM_DATA.

The evaluation manifest refers to each example by (source_file, source_row_index) plus
content hashes (image_sha256, question_sha256, source_shard_sha256). After downloading the
public datasets (docs/REPRODUCE.md), relocate the manifest so ``source_file`` points at
your local copies; the hashes guarantee the same examples are used.

Usage: python evaluation/tools/relocate_manifest.py IN OUT --map OLD_PREFIX=NEW_PREFIX [...]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path); ap.add_argument('dst', type=Path)
    ap.add_argument('--map', nargs='+', default=[], help='OLD=NEW prefix substitutions')
    a = ap.parse_args()
    maps = [m.split('=', 1) for m in a.map]
    d = json.loads(a.src.read_text())
    n = 0
    for rows in d['splits'].values():
        for r in rows:
            for key in ('source_file', 'image_path'):
                if key in r:
                    for old, new in maps:
                        if r[key].startswith(old):
                            r[key] = new + r[key][len(old):]; n += 1; break
    a.dst.write_text(json.dumps(d, indent=1))
    print(f'rewrote {n} paths -> {a.dst}')


if __name__ == '__main__':
    main()
