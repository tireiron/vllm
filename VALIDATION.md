# Validation evidence and reproduction

This is a review candidate against upstream vLLM commit
`3a90cc6fd0504b52bf47d58e8d70cff4b03f2c19`. Validation combines earlier GPU
numerical/reference and end-to-end serving checks with current-source CPU,
compiler and static checks. The tests and kernel benchmark generate numerical
tensors independently. The performance summary below contains aggregate timing
results only; application data, checkpoint identity and workload configuration
are not included.

## Measured benefit in the earlier release

The release-based SM120 implementation improved cold-prefill input-token
throughput in matched tests with the same request bytes, checkpoint and
speculative-decoding settings. Each row has one baseline/candidate pair.

| Cold-prefill case | Native input tok/s | Expanded input tok/s | Throughput gain | Prefill latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Shorter | 8,823 | 8,936 | 1.3% | 1.3% |
| Intermediate | 6,145 | 6,525 | 6.2% | 5.8% |
| Longest | 3,478 | 3,828 | 10.1% | 9.2% |

Rates are input tokens divided by the server's
`vllm:request_prefill_time_seconds_sum` counter delta, rounded to whole tok/s.
Percentages use unrounded values. All cases completed with zero prefix-cache
hits and no preemption. These are individual observations, not repeated-trial
means. The shortest case's small difference is not a demonstrated robust gain.
No output-token/decode speedup is attributed to the prefill patch.

Exact workload lengths, checkpoint identity, request contents and configuration
are withheld. Consequently this is a summary of earlier validation, not an
independently reproducible public-model result. The current-main port has not
been benchmarked; the public-model procedure below is a proposed follow-up.

## Measuring the benefit on other workloads

The optimization aims to lower prefill latency and time to first token when
native BF16-query/FP8-KV attention spends significant time converting the same
KV data for successive query tiles. It exchanges repeated conversions for a
page copy and temporary BF16 storage. Measure whether that exchange saves
time after including the copy, planning and allocation costs. Short prefills
and workloads limited by other operations may not improve.

Compare expansion disabled and enabled with all other settings held constant.
The decision is whether lower prefill/first-token latency outweighs the
memory cost without unacceptable decode, throughput, capacity or quality
regressions. A faster layer kernel alone is insufficient. All example sizes
and schedules below are independently chosen illustrations, separate from
the earlier measured cases above.

## Completed validation

Refreshed on October 31, 2026. All three contribution patches rebased
unchanged. CPU tests, interpreter cases, CLI checks and applicable pre-commit
hooks passed again using the existing isolated tools. Offline compilation
and changed-page documentation evidence are retained for unchanged sources.

| Check | Result and scope |
| --- | --- |
| Prior-release GPU validation | Passed numerical/reference and end-to-end serving checks; unchanged-helper comparison connects this evidence to the port |
| CPU page selection and budget behavior | 10 passed, 4 GPU cases deselected; source checkout and normal upstream conftest, with released runtime dependencies |
| Triton CPU interpreter | 48 layout/stride/page cases and 2 physical-offset cases passed; all 254 finite E4M3 encodings and addresses beyond 2 GiB; no CUDA context |
| Offline Triton compilation | Eight SM120 variants compiled to cubin: HND/NHD, page-index stride 1/2, int32/int64 indices; no GPU execution |
| Applicable pre-commit hooks | Passed for all six changed files, including Ruff, mypy, typos, Markdown lint and source/configuration checks |
| Python compilation | Passed for five changed Python files |
| CLI setting transport | Real serve parser and engine arguments accept 0 and 1 GiB, reject out-of-range values; no model loaded |
| CI selection | New test is selected by existing V1 Attention jobs and their source dependencies |
| Documentation | Changed page rendered successfully; fresh local discovery review completed; full site/public-preview checks pending |

CPU checks imported the actual source checkout with Python 3.12,
Torch 2.13.0+cu130 and Triton 3.7.1 from released runtime dependencies.

## Reusing prior validation

The earlier GPU execution is relevant to the unchanged helper: its host
control flow, metadata class, budget/threshold constants and feature call
expressions match the port after ignoring annotations/docstrings where
applicable. Eight compiled SM120 kernel instruction sections also match
byte for byte using Torch 2.13.0+cu130 and Triton 3.7.1. This carries forward
GPU evidence for those unchanged components and specializations.

## Coverage of the upstream port

The typed setting, exact SM120 eligibility gate, newer runner and pinned
dependency versions differ from the release-based integration. CLI transport
and bounds are checked, but this port has not had an SM120 serving rerun,
including performance, memory/concurrency and model evaluation, with the base
revision's dependencies. A dedicated test device is unavailable. Its four GPU
test cases also remain unrun locally; existing attention CI selects the helper
tests but cannot exercise the SM120-only serving path on other architectures.
CPU interpretation and offline compilation complement earlier GPU validation;
they do not execute the port's compiled GPU code or serving integration.

The review basis is the earlier validation, unchanged-helper comparison and
completed source checks. No current-main model-evaluation scores or serving
speedup numbers are asserted. The SM120 procedure below supports focused
follow-up if reviewers need further integration evidence before merge.

## Reproduce without a GPU

From the patched source checkout, in an isolated environment with the runtime
and test dependencies available, run the CPU selection:

```bash
CUDA_VISIBLE_DEVICES='' VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 VLLM_USE_RUST_FRONTEND=0 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -o addopts= -p no:cacheprovider \
  tests/v1/attention/test_fp8_prefill_cache.py -k prefill_metadata -q
```

Run the accompanying portable script from the same checkout, substituting its
location for the example path:

```bash
PYTHONPATH=. .venv/bin/python /path/to/validate_kernel_cpu.py
```

The script selects the CPU platform and disables CUDA visibility before
imports. It executes the unchanged gather/cast kernel through Triton's
[documented interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
An independent PyTorch oracle checks BF16 bits, including signed zeros,
shared/duplicate pages, strided K/V and indices, partial tiles, input
immutability and output sentinels. Two sparse anonymous mappings exercise
physical addresses beyond 2 GiB without allocating multi-GB physical buffers.
The script targets Linux and does not load a model, use CUDA, or exercise
FlashInfer attention, GPU allocation/capture or serving. `--output` optionally
saves a JSON report. It is a supplementary validation attachment, separate
from the proposed upstream commits.

## Focused GPU follow-up

To close the current-integration evidence gap when hardware becomes available
or a reviewer requests it, start with the focused GPU tests below, then a
public-model enabled/disabled comparison covering prefill and mixed prefill/
decode, including memory and model-quality checks. A broad search for optimal
serving settings is not required. Use an available SM120 test or CI worker.

Use an isolated upstream development environment with the dependencies pinned
by the base revision. Run the full relevant pre-commit hooks. On an available
SM120 GPU, run the new tests, then the generated-tensor benchmark:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_fp8_prefill_cache.py -v
.venv/bin/python benchmarks/kernels/benchmark_fp8_prefill_cache.py --layout NHD
.venv/bin/python benchmarks/kernels/benchmark_fp8_prefill_cache.py --layout HND
```

The benchmark reports the native path and gather/cast-plus-attention latency,
repeat count, range, median, scratch capacity and numerical difference. It
excludes metadata planning and is not an end-to-end serving benchmark.

For serving, use a public checkpoint supporting BF16 queries, E4M3 KV and
head dimension 256. A small functional-check candidate is
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json),
whose public configuration specifies head dimension 256 and context 262144.
It is a functional-check example, not a model recommendation or a substitute
for benchmarking larger serving models.
Keep checkpoint, scales, sampling, context limit, prefix
caching policy, scheduler limits and speculative settings identical in both
runs. Do not include private calibrations or application templates. Use
`--attention-backend FLASHINFER`; compare
`--attention-config '{"flashinfer_prefill_cache_expansion_bytes":0}'` with
`--attention-config '{"flashinfer_prefill_cache_expansion_bytes":1073741824}'`.
Check for `Using bounded FP8 cache expansion for native prefill` in the
enabled run. This confirms activation for at least one eligible batch;
later batches can still fall back, so verify the measured cases use expansion.

For example, after installing the isolated upstream development environment,
start the disabled arm on an unused loopback port:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_GPU_SYNC_CHECK=error \
  .venv/bin/vllm serve Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 8001 --language-model-only \
  --dtype bfloat16 --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER --max-model-len 131072 \
  --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --attention-config '{"flashinfer_prefill_cache_expansion_bytes":0}'
```

Run a generated cold batch, then repeat against the enabled arm after
restarting the isolated server with the budget changed to 1073741824:

```bash
.venv/bin/vllm bench serve --backend vllm \
  --model Qwen/Qwen3.5-0.8B --host 127.0.0.1 --port 8001 \
  --dataset-name random --random-input-len 16384 --random-output-len 128 \
  --num-prompts 32 --max-concurrency 8 --request-rate inf \
  --seed 2718 --temperature 0 --ignore-eos
```

These commands are a proposed reproduction procedure, not executed results.
Use prefix caching in both arms for a separate warmed-prefix experiment.

Use `VLLM_GPU_SYNC_CHECK=error` to check the asynchronous execution contract.
Fix the seed and output lengths. Repeat each selected baseline/candidate pair
at least three times, alternating order, and report median/range,
TTFT, TPOT, total time, admission,
preemptions, peak allocated/reserved memory and allocation retries/OOMs.
Use fresh cache state for cold-prefill runs. Synthetic throughput does not
establish model accuracy.

Run a public model evaluation with identical inputs and generation budgets
for both configurations; report accuracy and output-length differences. Add
real serving checks for shared prefixes, partial pages, mixed decode/prefill,
multiple live metadata objects and full-context retrieval. Cover the disabled
and unsupported-mode fallbacks, small prefills and over-budget batches.
Exercise the OOM fallback at the allocation boundary in a controlled test
environment; do not deliberately exhaust a production GPU.

The native BF16-query/non-unit-query-scale correctness issue is tracked by
vLLM PR #53754 and is independent of this optimization. Keep any necessary
correctness fix identical in both benchmark arms and disclose the exact base.

## Optional broader characterization

For claims spanning more workload sizes, expand the sweep with the public
`vllm bench serve --dataset-name random` generator. For example, sweep 1K,
4K, 16K and 64K input tokens at concurrency 1, 8 and 32, within the model's
context and memory limits. These are illustrative geometric test sizes,
not an optimized workload profile. Restrict any performance claim to the
configurations actually measured.
