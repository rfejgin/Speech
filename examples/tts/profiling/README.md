# EasyMagpie / Nemotron-H profiling

This recipe follows the `profiling_v3` run on `cs-oci-ord-vscode-02` (job 35801978).
The local starting commit `8fbcf75e26109a12c1aafa7c692978f6c3448011` differs from
that run's `222ae71956a60615fa98df2a1c2e42baa354ea91` only by the original 74-line
profiling addition in `easy_magpietts.py` (commits `73d11968a` and `222ae7195`).
The section names and capture controls are retained here, with their lifecycle
moved into a reusable session and Lightning profiler. Model math is unchanged.

`train.sub` is a **proposal**, retaining the original container, data pin,
model settings, batch sizes, 2 nodes, 8 GPUs/tasks per node, 12 CPUs per task,
4 training data-loader workers per rank, and 30-minute allocation. It has not
been submitted. No cluster files need to be edited to review this proposal.

## Two separate captures

| Setting | `PROFILE_MODE=light` (default) | `PROFILE_MODE=shapes` |
| --- | --- | --- |
| Warmup | 99 batches | 99 batches |
| Captured batches | 100–119 | 100–102 |
| Traced global ranks | 0 and 8 | 0 and 8 |
| CUDA / NVTX / OS runtime / NCCL | Yes | Yes |
| Per-operator autograd shapes | No | Yes; higher overhead |
| Python stack sampling | Off; optionally 200 Hz | Off; optionally 200 Hz |

Use the light run for time attribution; prefer interior iterations 101–118 when
comparing timings to reduce capture-boundary effects. Use the shapes run to identify operator
sizes and forward/backward relationships. Do not compare their throughput as an
optimization result: shapes instrumentation adds substantial overhead. Do not
combine this with a CUDA-enabled `torch.profiler` / Lightning `pytorch` profiler.
The previous Lightning `simple` profiler is replaced by the NVTX adapter.

These settings can be exported before using the existing `submit_chain.sh`:

```bash
export SPEECH_COMMIT_ID=<published-40-character-SHA-containing-these-changes>
export PROFILE_MODE=light
export PROFILE_RANKS=0,8
export PROFILE_START_STEP=100
# Optional: PROFILE_NUM_STEPS=20, PROFILE_PYTHON_SAMPLING=true
```

This is preparation only, not a launch command. Commit and publish the reviewed
code before setting `SPEECH_COMMIT_ID`; the old trace SHA does not include the new
instrumentation. Place the proposed `.sub` in a fresh experiment directory when
preparing a future submission through the original launcher. `--export=ALL` in
that launcher passes the profiling controls through. Debug mode is rejected
because its allocation and batch limits differ; fewer nodes require an explicit
matching `PROFILE_RANKS` choice and a separately reviewed resource request.

Nsight in the container must support `--trace=nccl`, `--python-sampling`, and,
for shapes mode, `--pytorch=autograd-shapes-nvtx`. Missing Nsight or unsupported
options fail the launch instead of silently producing an unprofiled run. The
script prints its version. No packages or profiling tools are installed by it.
CPU context-switch tracing is disabled because the earlier run lacked permission.

## What the trace and metadata contain

- `iteration/N` spans the main process's data-fetch boundary through transfer,
  forward, backward, optimizer execution, callbacks and logging, ending just
  before the next data fetch. Data-loader workers can prefetch in parallel;
  this range measures the main process's wait, not all worker preparation time.
  Lightning prefetching can fetch the following batch at this boundary.
- The original `codec_context`, `codec_target`, `codec_user`, `user_turn_loop`,
  `process_batch` and `metric_logging` ranges remain.
- Backbone layers include their full module path and layer type. MoE layers
  **1, 15 and 30** additionally separate routing, one-hot creation, per-expert
  dispatch, expert compute, combine, and empty-expert no-op computation.
  Shared experts have their own range. Dispatch here is local expert routing;
  it does not introduce an MoE all-to-all communication operation.
- Codebook, local-transformer, acoustic-predictor and phoneme losses have separate
  ranges. Each acoustic predictor block includes its own projection and loss.
- Lightning actions label backward, device transfer, and optimizer execution;
  gradient clipping is separately annotated. **Lightning's optimizer-step range
  includes the closure (forward/backward/clipping).** Do not treat its inclusive
  duration as parameter-update time or add it to its nested ranges.
- JSON records contain batch/global-step/epoch IDs, dataset and task labels,
  tensor shapes, valid lengths, selected training mode, and per-expert assignment
  counts. Counts include padded backbone positions and should be interpreted
  alongside `backbone.valid_lengths`; their sum is padded tokens × top-k.
  Empty counts expose the dummy expert path. Recomputed MoE forwards under
  activation checkpointing produce additional records/ranges.

GPU length tensors are small detached clones, labeled `profiling/metadata`; CPU copies and JSON writes happen
**after** `cudaProfilerStop`. Expert counts use the shapes already returned by
`torch.where`, without an additional GPU histogram or host readback. No audio,
text token contents, logits, or autograd graphs are saved. Records are capped
(default 10,000), with a dropped-record count in the sidecar.

Reports are written under `<experiment>/nsys/<job-id>/<mode>/rank<rank>.nsys-rep`
and sidecars under `metadata/rank<rank>.json`. Slurm logs use global task IDs
(`%t`) so eight ranks do not share one node's output file. The resolved Hydra
configuration and code/data commit IDs remain in the training logs.

Compare the same iteration IDs across ranks for NCCL waiting and stragglers;
absolute timestamps on different hosts may need clock alignment. Two ranks give
one view per node, not visibility into every possible straggler. First check
Nsight diagnostics for dropped CUDA/NVTX records before aggregating durations.

## Capture semantics and configuration

The existing model options remain opt-in:

```text
++model.profile_sections=true
++model.profile_sections_sync=false
++model.profile_sections_interval=50
++model.nsys_profile_start_step=100
++model.nsys_profile_num_steps=20
'++model.nsys_profile_ranks=[0,8]'
'++model.profile_moe_layers=[1,15,30]'
++model.profile_metadata_dir=/exp_dir/nsys/<job>/<mode>/metadata
```

The entrypoint `examples/tts/easy_magpietts.py` binds the custom Lightning profiler.
For another entrypoint, construct `TTSLightningProfiler`, pass it to the Trainer,
bind the model before `fit`, and call `model._tts_profile.finish()` in `finally`.
Step numbers are one-based training batches observed in this fit invocation,
not resumed optimizer steps; gradient accumulation does not change this count.
Use the normal automatic-optimization training loop, not a custom
`training_step(dataloader_iter)` that fetches multiple batches per step.

A CUDA synchronization drains warmup before capture and final work before stop;
the latter is labeled `capture_final_drain`. There are no per-section CUDA
synchronizations in the proposed run. Reported section times are **host wall
clock / enqueue times**, not GPU durations; use the GPU timeline for GPU time.
`profile_sections_sync=true` is still available for intrusive timing without an
Nsight capture, and is rejected when a capture window is configured. Timing
summaries report means per call, not the original fixed-interval approximation.

The proposed run disables validation/sanity batches/checkpointing, uses a fresh
output directory with no resume, and finishes five batches after the capture
window. Its 20-minute training timer leaves room inside the 30-minute allocation.
If training stops early, the session closes in `on_train_end` or the entrypoint's
`finally` block. A hard process kill cannot guarantee report or metadata recovery.

## Validation

Dependency-free lifecycle, rank selection, exception cleanup, deferred copies,
record limits, and mocked launcher tests (plus a real CPU Trainer test when
Lightning is installed):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
  -o addopts='' tests/collections/tts/parts/utils/test_tts_profiling.py
bash -n examples/tts/profiling/train.sub
shellcheck examples/tts/profiling/train.sub
```

In an existing NeMo test environment, also run the numerical MoE equivalence test
(forward, input/parameter gradients, and empty experts):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest --cpu \
  tests/collections/tts/test_nemotron_h_decoder.py -k profiling_preserves
```

The analysis host passed 24 dependency-free tests. It lacks PyTorch/Lightning,
so the real Trainer test was skipped, and the numerical MoE
test, and CUDA trace validation require an existing suitable environment. Nothing
is installed or submitted as part of preparing these changes.

Black, Python parsing, Bash syntax, ShellCheck, and `git diff --check` passed.
The full pre-commit suite was not run because pre-commit/isort are unavailable
on this host; no installation was attempted.
