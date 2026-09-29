# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightning's profiler interface supplies data-fetch and training-loop boundaries."""

from collections import defaultdict

from lightning.pytorch.profilers import Profiler


class TTSLightningProfiler(Profiler):
    """Translate Lightning actions to NVTX without enabling a second CUDA collector.

    Lightning 2.4 emits ``*.train_dataloader_next`` before the fetch/device
    transfer. Iterations run from one such boundary to the next. Optimizer-step
    actions include their closure; nested forward/backward/clipping ranges must
    be excluded when interpreting optimizer time.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self._ranges = defaultdict(list)

    def bind(self, model):
        self.model = model

    def start(self, action_name):
        if self.model is None:
            return
        session = self.model._tts_profile
        if action_name.endswith('.on_train_end'):
            session.finish()
        if action_name.endswith('.train_dataloader_next'):
            session.begin_iteration(rank=int(self.model.global_rank))
        # Epoch/run scopes cross iteration boundaries and would break the NVTX
        # stack when a capture begins or ends inside them. Keep only batch scopes.
        if not (
            action_name == 'run_training_batch'
            or action_name.rsplit('.', 1)[-1]
            in {
                'train_dataloader_next',
                'batch_to_device',
                'training_step',
                'backward',
                'optimizer_step',
                'optimizer_zero_grad',
                'on_train_batch_start',
                'on_train_batch_end',
                'on_before_backward',
                'on_after_backward',
                'on_before_optimizer_step',
                'configure_gradient_clipping',
            }
        ):
            return
        context = session.range(f'lightning/{action_name}')
        context.__enter__()
        self._ranges[action_name].append(context)

    def stop(self, action_name):
        contexts = self._ranges.get(action_name)
        if contexts:
            contexts.pop().__exit__(None, None, None)
