# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free tests; may also run with pytest --noconftest on an analysis host."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiling = load_module('tts_profiling', 'nemo/collections/tts/parts/utils/tts_profiling.py')


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.events = []
        self.stack = []
        self.capturing = False
        self.profiler = SimpleNamespace(start=self.start, stop=self.stop)
        self.nvtx = SimpleNamespace(range_push=self.push, range_pop=self.pop)

    def is_available(self):
        return self.available

    def synchronize(self):
        self.events.append('sync')

    def start(self):
        assert not self.capturing
        self.capturing = True
        self.events.append('start')

    def stop(self):
        assert self.capturing
        assert not self.stack
        self.capturing = False
        self.events.append('stop')

    def push(self, name):
        self.stack.append(name)
        self.events.append(('push', name))

    def pop(self):
        self.events.append(('pop', self.stack.pop()))


class FakeTensor:
    def __init__(self, values, cuda):
        self.values = list(values)
        self.shape = (len(values),)
        self.cuda = cuda

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values, self.cuda)

    def cpu(self):
        assert not self.cuda.capturing, 'Metadata copied to CPU inside the capture'
        self.cuda.events.append('cpu')
        return self

    def tolist(self):
        return self.values


def session(cuda, **overrides):
    cfg = dict(profile_sections=True, nsys_profile_start_step=2, nsys_profile_num_steps=2)
    cfg.update(overrides)
    return profiling.TTSProfileSession(cfg, cuda=cuda)


def test_capture_covers_exact_batches_and_drains_only_at_boundaries():
    cuda = FakeCuda()
    profile = session(cuda)
    profile.begin_iteration()
    assert not cuda.events
    profile.begin_iteration()
    with profile.range('forward'):
        with profile.range('expert'):
            assert cuda.stack == ['iteration/2', 'forward', 'expert']
    profile.begin_iteration()
    assert cuda.capturing
    profile.begin_iteration()
    assert not cuda.capturing
    assert not cuda.stack
    assert [event for event in cuda.events if isinstance(event, str)] == ['sync', 'start', 'sync', 'stop']
    assert ('push', 'iteration/2') in cuda.events
    assert ('push', 'iteration/3') in cuda.events
    assert ('push', 'iteration/4') not in cuda.events
    profile.finish()
    profile.begin_iteration()
    assert cuda.events.count('start') == cuda.events.count('stop') == 1


@pytest.mark.parametrize('rank', [1, 7, 9])
def test_unselected_global_ranks_do_no_cuda_work(rank):
    cuda = FakeCuda()
    profile = session(cuda, nsys_profile_ranks=[0, 8])
    for _ in range(5):
        profile.begin_iteration(rank=rank)
        with profile.range('work'):
            pass
    profile.finish()
    assert not cuda.events


def test_rank_eight_is_selected():
    cuda = FakeCuda()
    profile = session(cuda, nsys_profile_ranks=[0, 8])
    profile.begin_iteration(rank=8)
    profile.begin_iteration(rank=8)
    assert cuda.capturing
    profile.finish()


def test_disabled_session_does_not_import_or_touch_cuda():
    cuda = FakeCuda()
    profile = profiling.TTSProfileSession({}, cuda=cuda)
    profile.begin_iteration()
    with profile.range('work'):
        pass
    profile.finish()
    assert not cuda.events


def test_exception_balances_ranges_and_finish_stops_last_batch():
    cuda = FakeCuda()
    profile = session(cuda, nsys_profile_start_step=1)
    profile.begin_iteration()
    with pytest.raises(RuntimeError, match='training failed'):
        with profile.range('forward'):
            with profile.range('expert'):
                raise RuntimeError('training failed')
    assert cuda.stack == ['iteration/1']
    profile.finish()
    assert not cuda.stack and not cuda.capturing


def test_metadata_is_bounded_detached_and_written_after_stop(tmp_path):
    cuda = FakeCuda()
    profile = session(cuda, profile_metadata_dir=str(tmp_path), profile_max_records=2)
    tensor = FakeTensor([8, 12], cuda)
    profile.begin_iteration()
    profile.record_tensors('warmup', {'audio_lens': tensor})
    profile.begin_iteration()
    profile.record_tensors('batch', {'audio_lens': tensor, 'audio': tensor}, dataset_names=['test'])
    tensor.values[0] = 999
    profile.record('moe', expert_assignments=[4, 4, 4, 4])
    profile.record('overflow')
    assert not (tmp_path / 'rank0.json').exists()
    assert 'cpu' not in cuda.events
    profile.finish()
    result = json.loads((tmp_path / 'rank0.json').read_text())
    assert result['dropped_records'] == 1
    assert len(result['records']) == 2
    assert result['records'][0]['lengths'] == {'audio_lens': [8, 12]}
    assert result['records'][0]['shapes'] == {'audio_lens': [2], 'audio': [2]}
    assert cuda.events.index('stop') < cuda.events.index('cpu')
    assert not profile._records


@pytest.mark.parametrize(
    'overrides',
    [
        {'nsys_profile_start_step': 0},
        {'nsys_profile_start_step': 1.5},
        {'nsys_profile_num_steps': 0},
        {'profile_max_records': 0},
        {'profile_sections_interval': 0},
        {'nsys_profile_ranks': [-1]},
        {'profile_moe_layers': [-1]},
        {'profile_sections_sync': True},
    ],
)
def test_invalid_capture_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        session(FakeCuda(), **overrides)


def test_missing_cuda_fails_when_capture_is_requested():
    profile = session(FakeCuda(available=False), nsys_profile_start_step=1)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        profile.begin_iteration()


def test_intrusive_section_timing_remains_available_without_capture():
    cuda = FakeCuda()
    profile = session(cuda, nsys_profile_start_step=None, profile_sections_sync=True)
    profile.begin_iteration()
    with profile.range('codec'):
        pass
    profile.finish()
    assert cuda.events.count('sync') == 4
    assert 'start' not in cuda.events


def test_lightning_adapter_excludes_scopes_crossing_capture_boundaries(monkeypatch):
    fake_lightning = ModuleType('lightning.pytorch.profilers')
    fake_lightning.Profiler = object
    monkeypatch.setitem(sys.modules, 'lightning.pytorch.profilers', fake_lightning)
    bridge = load_module('tts_lightning_profiler', 'nemo/collections/tts/parts/utils/tts_lightning_profiler.py')
    cuda = FakeCuda()
    profile = session(cuda, nsys_profile_start_step=1, nsys_profile_num_steps=1)
    adapter = bridge.TTSLightningProfiler()
    adapter.bind(SimpleNamespace(_tts_profile=profile, global_rank=0))
    adapter.start('run_training_epoch')
    adapter.start('[_TrainingEpochLoop].train_dataloader_next')
    adapter.stop('[_TrainingEpochLoop].train_dataloader_next')
    adapter.start('run_training_batch')
    adapter.start('[Strategy]DDPStrategy.backward')
    adapter.stop('[Strategy]DDPStrategy.backward')
    adapter.stop('run_training_batch')
    adapter.stop('run_training_epoch')
    adapter.start('[Callback]Timer.on_train_end')
    adapter.stop('[Callback]Timer.on_train_end')
    assert not cuda.stack
    assert cuda.events.count('stop') == 1
    assert ('push', 'lightning/run_training_epoch') not in cuda.events
    assert ('push', 'lightning/[Strategy]DDPStrategy.backward') in cuda.events


@pytest.mark.parametrize('rank,mode,traced', [(0, 'light', True), (8, 'shapes', True), (1, 'light', False)])
def test_rank_wrapper_and_shape_mode_without_launching_training(rank, mode, traced):
    # Evaluate only the rank-selection fragment, with nsys/mkdir/python replaced
    # by shell functions. Never source the submission file or invoke Slurm.
    source = (ROOT / 'examples/tts/profiling/train.sub').read_text()
    wrapper = source[source.index('# The same batch window') : source.index('  --config-name=')]
    wrapper = wrapper.replace('\\${', '${').replace('\\\n', '')
    script = (
        '''set -eu
nsys() { printf 'NSYS'; printf ' <%s>' "$@"; printf '\\n'; }
python() { printf 'PYTHON'; printf ' <%s>' "$@"; printf '\\n'; }
mkdir() { :; }
'''
        + wrapper
        + '\n'
    )
    result = subprocess.run(
        ['bash', '-c', script],
        text=True,
        capture_output=True,
        timeout=5,
        env=dict(
            PATH='/usr/bin:/bin',
            SLURM_PROCID=str(rank),
            SLURM_JOB_ID='123',
            PROFILE_RANKS='0,8',
            PROFILE_MODE=mode,
            PROFILE_PYTHON_SAMPLING='false',
        ),
    )
    assert result.returncode == 0, result.stderr
    assert ('NSYS <profile>' in result.stdout) == traced
    assert ('PYTHON <examples/tts/easy_magpietts.py>' in result.stdout) != traced
    assert ('--pytorch=autograd-shapes-nvtx' in result.stdout) == (traced and mode == 'shapes')
    if traced:
        assert f'/rank{rank}>' in result.stdout


def test_submission_heredoc_leaves_rank_expansion_to_each_task():
    import re

    source = (ROOT / 'examples/tts/profiling/train.sub').read_text()
    start = source.index('cmd=$(cat <<EOF_CMD')
    end = source.index('\n)\n', start) + 3
    fragment = source[start:end] + '\nprintf "%s" "$cmd"\n'
    # Expand only the heredoc, not the submit script. All values are synthetic;
    # unbound per-task variables catch accidental expansion on the batch host.
    names = set(re.findall(r'\$\{([A-Z_]+)', fragment))
    environment = {name: 'fixture' for name in names - {'SLURM_PROCID', 'NSYS_PREFIX'}}
    environment.update(PATH='/usr/bin:/bin', PROFILE_RANKS='0,8', PROFILE_MODE='light')
    result = subprocess.run(
        ['bash', '-eu', '-c', fragment], env=environment, text=True, capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert '"${NSYS_PREFIX[@]}" python' in result.stdout
    assert '/rank${SLURM_PROCID}' in result.stdout
    assert "'++model.nsys_profile_ranks=[0,8]'" in result.stdout
    syntax = subprocess.run(['bash', '-n'], input=result.stdout, text=True, capture_output=True, timeout=5)
    assert syntax.returncode == 0, syntax.stderr


def test_real_lightning_fetch_and_optimizer_boundaries(tmp_path):
    pl = pytest.importorskip('lightning.pytorch')
    torch = pytest.importorskip('torch')
    from torch.utils.data import DataLoader, TensorDataset

    bridge = load_module('tts_lightning_integration', 'nemo/collections/tts/parts/utils/tts_lightning_profiler.py')
    cuda = FakeCuda()
    profile = session(cuda, profile_metadata_dir=str(tmp_path))

    class TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self._tts_profile = profile
            self.linear = torch.nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            profile.record('batch', batch_idx=batch_idx, global_step=self.global_step)
            return self.linear(batch[0]).square().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    adapter = bridge.TTSLightningProfiler()
    model = TinyModel()
    adapter.bind(model)
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_steps=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        profiler=adapter,
        default_root_dir=str(tmp_path),
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=1))
    finally:
        profile.finish()
        torch.set_num_threads(threads)
    metadata = json.loads((tmp_path / 'rank0.json').read_text())
    assert [(row['iteration'], row['batch_idx'], row['global_step']) for row in metadata['records']] == [
        (2, 1, 1),
        (3, 2, 2),
    ]
    assert any(name.endswith('.backward') for name in metadata['ranges'])
    assert cuda.events.count('start') == cuda.events.count('stop') == 1
    assert not cuda.stack


def test_early_exit_uses_global_rank_before_first_fetch(monkeypatch, tmp_path):
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.setenv('SLURM_PROCID', '8')
    profile = session(FakeCuda(), nsys_profile_ranks=[0, 8], profile_metadata_dir=str(tmp_path))
    profile.finish()
    assert (tmp_path / 'rank8.json').exists()
    assert not (tmp_path / 'rank0.json').exists()
