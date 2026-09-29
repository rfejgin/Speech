# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in NVTX ranges and bounded, deferred metadata for TTS training.

This module deliberately has no eager torch/Lightning imports so capture lifecycle
and metadata handling can be tested without initializing CUDA.
"""

import json
import logging
import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

_LOG = logging.getLogger(__name__)
_NO_RANGE = nullcontext()


class TTSProfileSession:
    """One capture per process; step numbers count training batches since fit began.

    ``begin_iteration`` runs immediately before Lightning's training data fetch.
    The next fetch closes the preceding iteration, including backward/optimizer
    and end-of-batch logging. CUDA is synchronized only at capture boundaries
    unless the explicitly intrusive ``profile_sections_sync`` option is enabled.
    """

    def __init__(self, cfg, cuda=None):
        self.sections = bool(cfg.get('profile_sections', False))
        self.start_step = cfg.get('nsys_profile_start_step')
        self.num_steps = int(cfg.get('nsys_profile_num_steps', 5))
        self.ranks = tuple(int(rank) for rank in cfg.get('nsys_profile_ranks', [0]))
        self.sync = bool(cfg.get('profile_sections_sync', False))
        self.interval = int(cfg.get('profile_sections_interval', 50))
        self.metadata_dir = cfg.get('profile_metadata_dir')
        self.max_records = int(cfg.get('profile_max_records', 10000))
        self.moe_layers = tuple(int(layer) for layer in cfg.get('profile_moe_layers', [1]))
        self.enabled = self.sections or self.start_step is not None
        if self.start_step is not None and (int(self.start_step) != self.start_step or self.start_step < 1):
            raise ValueError('nsys_profile_start_step must be a positive, one-based batch number')
        if min(self.num_steps, self.interval, self.max_records) < 1:
            raise ValueError('profile step counts, interval and record limit must be positive')
        if any(rank < 0 for rank in self.ranks) or any(layer < 0 for layer in self.moe_layers):
            raise ValueError('profile ranks and layer indices must be nonnegative')
        if self.start_step is not None and self.sync:
            raise ValueError('Use profile_sections_sync=false for an Nsight capture')
        if cuda is None and self.enabled:
            import torch

            cuda = torch.cuda
        self.cuda = cuda
        self.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
        self._use_cuda = False
        self.iteration = 0
        self.capturing = False
        self.finished = False
        self._iteration_range = None
        self._records = []
        self._dropped = 0
        self._timings = defaultdict(lambda: [0.0, 0])

    @property
    def active(self):
        return (
            self.enabled
            and self.iteration > 0
            and not self.finished
            and self.rank in self.ranks
            and (self.capturing or self.start_step is None)
        )

    @property
    def recording_metadata(self):
        return self.active and self.metadata_dir is not None

    def begin_iteration(self, rank=0):
        self._end_iteration()
        self.rank = rank
        self.iteration += 1
        if not self.enabled or self.finished or self.rank not in self.ranks:
            return
        if self.capturing and self.iteration >= self.start_step + self.num_steps:
            self.finish()
            return
        self._use_cuda = self.cuda is not None and self.cuda.is_available()
        if self.start_step is not None and self.iteration == self.start_step:
            if not self._use_cuda:
                raise RuntimeError('An Nsight capture requires CUDA')
            self.cuda.synchronize()  # Drain the warmup before capture, not inside the measured iterations.
            self.cuda.profiler.start()
            self.capturing = True
            _LOG.info('[nsys] capture start: rank=%s batch=%s', self.rank, self.iteration)
        if self.active:
            self._iteration_range = self.range(f'iteration/{self.iteration}', force=True)
            self._iteration_range.__enter__()

    def range(self, name, force=False):
        if not self.active or not (self.sections or force):
            return _NO_RANGE
        return self._range(name)

    def record(self, kind, **values):
        if not self.recording_metadata:
            return
        if len(self._records) >= self.max_records:
            self._dropped += 1
            return
        self._records.append(dict(kind=kind, iteration=self.iteration, **values))

    def record_tensors(self, kind, tensors, **values):
        """Read shapes on the host; retain only small detached length/mask tensors.

        Values are transferred to the host after cudaProfilerStop. No input audio,
        token IDs, logits or autograd graphs are retained.
        """
        if not self.recording_metadata:
            return
        if len(self._records) >= self.max_records:
            self._dropped += 1
            return
        with self.range('profiling/metadata'):
            shapes, lengths = {}, {}
            for key, tensor in tensors.items():
                if not hasattr(tensor, 'shape'):
                    continue
                shapes[key] = list(tensor.shape)
                if key.endswith(('_lens', '_lengths')) and tensor.numel() <= 4096:
                    lengths[key] = tensor.detach().clone()
            self.record(kind, shapes=shapes, lengths=lengths, **values)

    def finish(self):
        if self.finished:
            return
        self._end_iteration()
        try:
            if self.capturing:
                # Account for asynchronous work from the final batch before stopping.
                with self.range('capture_final_drain', force=True):
                    self.cuda.synchronize()
                self.cuda.profiler.stop()
                self.capturing = False
                _LOG.info('[nsys] capture stop: rank=%s', self.rank)
        finally:
            self.finished = True
        if self.enabled and self.rank in self.ranks:
            if self.start_step is not None and self.iteration < self.start_step:
                _LOG.warning('[nsys] training ended before the requested capture window')
            self._write_metadata()
            self._report()

    @contextmanager
    def _range(self, name):
        use_cuda = self._use_cuda
        if use_cuda:
            if self.sync:
                self.cuda.synchronize()
            self.cuda.nvtx.range_push(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                if self.sync:
                    self.cuda.synchronize()
                self.cuda.nvtx.range_pop()
            totals = self._timings['iteration' if name.startswith('iteration/') else name]
            totals[0] += time.perf_counter() - started
            totals[1] += 1

    def _end_iteration(self):
        if self._iteration_range is not None:
            self._iteration_range.__exit__(None, None, None)
            self._iteration_range = None
            if self.start_step is None and self.iteration % self.interval == 0:
                self._report()

    def _report(self):
        if self._timings:
            mode = 'synchronized_wall' if self.sync else 'host_wall_including_enqueue'
            summary = ' '.join(
                f'{name}={total / count:.6f}s/{count}calls'
                for name, (total, count) in sorted(self._timings.items())
                if not name.startswith(('iteration/', 'lightning/'))
            )
            _LOG.info('[section_profile] rank=%s timing=%s mean_per_call: %s', self.rank, mode, summary)

    def _write_metadata(self):
        if self.metadata_dir is None:
            self._records.clear()
            return
        directory = Path(self.metadata_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'rank{self.rank}.json'
        payload = {
            'schema_version': 1,
            'rank': self.rank,
            'host': socket.gethostname(),
            'pid': os.getpid(),
            'capture_start_batch': self.start_step,
            'capture_num_batches': self.num_steps,
            'dropped_records': self._dropped,
            'timing_kind': 'synchronized_wall' if self.sync else 'host_wall_including_enqueue',
            'ranges': {name: {'seconds': total, 'calls': count} for name, (total, count) in self._timings.items()},
            'records': self._records,
        }
        with path.open('w') as stream:
            json.dump(payload, stream, default=lambda tensor: tensor.cpu().tolist(), indent=2)
            stream.write('\n')
        self._records.clear()
        _LOG.info('[nsys] metadata written to %s', path)


def module_profile_range(module, phase, detail=False):
    """Use the module's full path so predictor blocks cannot alias backbone layers."""
    session = getattr(module, '_tts_profile', None)
    if session is None or not session.active or (detail and not getattr(module, '_tts_profile_detail', False)):
        return _NO_RANGE
    return session.range(f'{module._tts_profile_name}/{phase}')
