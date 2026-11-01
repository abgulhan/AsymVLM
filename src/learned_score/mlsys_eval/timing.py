"""Stage-inclusive request timing; measured totals are never sums of estimates."""

import time
from collections import defaultdict
from contextlib import contextmanager


STAGES = frozenset({"queue", "preprocess", "vision", "selector", "prefill", "decode", "compression", "postprocess"})


class TimingContract:
    """Start before processor work; pass arrival_time to include server queuing.

    ``synchronize`` should be ``torch.cuda.synchronize`` for isolated GPU latency
    measurements. Do not use device-wide synchronization in a concurrent serving
    benchmark: that changes the scheduler. Mark that mode with ``serving=True``;
    stage values then measure host intervals, not exclusive GPU kernel time.
    """

    def __init__(self, *, synchronize=None, clock=time.perf_counter,
                 arrival_time=None, serving=False):
        if serving and synchronize is not None:
            raise ValueError("device synchronization is invalid for concurrent serving timing")
        self.clock, self.synchronize, self.serving = clock, synchronize, serving
        if synchronize:
            synchronize()
        now = clock()
        self.start = now if arrival_time is None else float(arrival_time)
        if self.start > now:
            raise ValueError("arrival time cannot be in the future")
        self.stages = defaultdict(float)
        self.stages["queue"] = now - self.start
        self.first_token = None
        self.end = None
        self.active = None

    @contextmanager
    def stage(self, name):
        if name not in STAGES:
            raise ValueError(f"unknown timing stage: {name}")
        if self.active is not None or self.end is not None:
            raise ValueError("stages must be non-overlapping and precede finish")
        if self.synchronize:
            self.synchronize()
        begin = self.clock()
        self.active = name
        try:
            yield
        finally:
            if self.synchronize:
                self.synchronize()
            self.stages[name] += self.clock() - begin
            self.active = None

    def mark_first_token(self):
        if self.first_token is not None or self.end is not None:
            raise ValueError("first-token marker must occur exactly once before finish")
        if self.synchronize:
            self.synchronize()
        self.first_token = self.clock()

    def finish(self):
        if self.active is not None or self.end is not None:
            raise ValueError("cannot finish inside a stage or twice")
        if self.synchronize:
            self.synchronize()
        self.end = self.clock()
        elapsed = self.end - self.start
        return {"end_to_end_seconds": elapsed,
                "ttft_seconds": None if self.first_token is None else self.first_token - self.start,
                "queue_seconds": self.stages["queue"],
                "compression_seconds": self.stages["compression"],
                "stage_seconds": {name: self.stages[name] for name in sorted(STAGES)},
                "unattributed_seconds": elapsed - sum(self.stages.values()),
                "timing_contract": "arrival_through_completed_output_v1",
                "timing_mode": "concurrent_host_intervals" if self.serving else "isolated_synchronized" if self.synchronize else "host_only",
                "includes_preprocessing": True,
                "file_loading_included": False}
