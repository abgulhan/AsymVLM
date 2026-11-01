"""Per-retention VisionZip execution for Qwen2.5-VL.

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
import argparse

import fcntl

import importlib.util

import json

import os

from pathlib import Path

import re

import signal

import socket

import subprocess

import threading

import time

import traceback

from learned_score.mlsys_sprint import manuscript_campaign_v35 as base

from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime, selector

from learned_score.mlsys_sprint.manuscript_data_v35 import family

from learned_score.mlsys_sprint.state import atomic_json, digest, file_digest

from learned_score.mlsys_sprint.followup_common import write_immutable

def visionzip_physical(runtime, entry, percent, root, addons):
    original=runtime.config
    raw,prepared,feature=runtime.geometry(entry)
    del raw,prepared
    count=(feature['input_visual_tokens']*percent+99)//100
    old_path=Path(original['output_root'])/'addons/records'/runtime.model/family(entry['dataset'])/'visionzip_native'/(entry['row_hash']+'.json')
    if old_path.exists():
        old=base.read(old_path)
        if (old.get('status')=='ok' and old.get('bundle_hash')==original['bundle_hash']
                and old.get('source_sha256')==file_digest(addons.__file__)
                and old.get('retained_visual_tokens')==count
                and old.get('input_visual_tokens')==feature['input_visual_tokens']):
            physical=base.read(old['raw_path'])
            if not physical.get('physical_measurement_performed') or physical.get('cache_key')!=old['cache_key']:
                raise ValueError('invalid existing physical VisionZip prediction')
            return {**old,'percent':percent,'panel_order':entry['panel_order'],'reused_from':str(old_path)}
    # Isolate each retention: the legacy addon's record filename omitted count.
    runtime.config={**original,'output_root':str(root/'visionzip'/str(percent))}
    try:
        value,_=addons.physical(runtime,entry,'visionzip_native',count)
    finally:
        runtime.config=original
    return {**value,'percent':percent,'panel_order':entry['panel_order']}


