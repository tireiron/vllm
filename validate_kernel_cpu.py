# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check prefill page conversion and indexing with Triton's CPU interpreter.

Uses Triton's documented interpreter mode, real tensors and independent
PyTorch casts/indexing as the oracle. It does not mock vLLM or exercise CUDA,
FlashInfer attention, GPU allocation or serving integration.
"""

import argparse
import gc
import hashlib
import itertools
import json
import mmap
import os
import resource
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["TRITON_INTERPRET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import triton

from vllm.v1.attention.ops import fp8_prefill_cache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Optional JSON result path")
args = parser.parse_args()
_expand_pages = fp8_prefill_cache._expand_pages

torch.set_num_threads(1)
assert type(_expand_pages).__name__ == "InterpretedFunction"
assert not torch.cuda.is_initialized()
codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
finite_codes = codes[torch.isfinite(codes.view(torch.float8_e4m3fn).float())]
assert len(finite_codes) == 254


def check(layout, page_size, storage_kind, index_dtype, index_stride):
    pages, heads, dim = 9, 3, 256
    shape = (
        (pages, heads, page_size, dim * 2)
        if layout == "HND"
        else (pages, page_size, heads, dim * 2)
    )
    count = 1
    for size in shape:
        count *= size
    raw = finite_codes[torch.arange(count + 7) % len(finite_codes)]
    storage = raw[7:].view(torch.float8_e4m3fn).reshape(shape)
    if storage_kind == "split":
        keys, values = storage[..., :dim], storage[..., dim:]
    else:
        keys, values = storage[..., ::2], storage[..., 1::2]
    original = raw.clone()
    logical_pages = torch.tensor([7, 0, 7, 3], dtype=index_dtype)
    index_backing = torch.full((2 + 4 * index_stride,), 8, dtype=index_dtype)
    indices = index_backing[1 : 1 + 4 * index_stride : index_stride]
    indices.copy_(logical_pages)
    result_shape = (4, 2, *keys.shape[1:])
    elements = 4 * 2 * heads * page_size * dim
    guard = 31
    output_backing = torch.full((elements + guard * 2,), 123.0, dtype=torch.bfloat16)
    output = output_backing[guard:-guard].reshape(result_shape)
    hnd_keys = keys if layout == "HND" else keys.transpose(1, 2)
    hnd_values = values if layout == "HND" else values.transpose(1, 2)
    _expand_pages[(4, triton.cdiv(heads * page_size * dim, 2048))](
        hnd_keys,
        hnd_values,
        indices,
        output,
        indices.stride(0),
        *hnd_keys.stride(),
        *hnd_values.stride(),
        heads,
        page_size,
        dim,
        layout == "NHD",
        2048,
        num_warps=4,
    )
    for component, source in enumerate((keys, values)):
        expected = source.bfloat16()[logical_pages.long()]
        # Integer bits also check signed zero, unlike floating-point closeness.
        assert torch.equal(
            output[:, component].view(torch.int16), expected.view(torch.int16)
        )
    assert torch.equal(raw, original), "Persistent input bytes changed"
    assert torch.equal(indices, logical_pages), "Page metadata changed"
    assert bool(torch.all(output_backing[:guard] == 123))
    assert bool(torch.all(output_backing[-guard:] == 123))
    return {
        "layout": layout,
        "page_size": page_size,
        "storage": storage_kind,
        "index_dtype": str(index_dtype),
        "index_stride": index_stride,
        "passed": True,
    }


def check_large_offset(index_dtype):
    # Sparse anonymous mappings reserve address space, not multi-GB physical
    # buffers. Only two short rows are touched. CPU interpreter storage copies
    # retain these same CPU storages, so no CUDA or bulk copy is involved.
    dim = 256
    physical = (1 << 31) // dim + 3
    pages = physical + 1
    mappings = [mmap.mmap(-1, pages * dim) for _ in range(2)]
    raw = [torch.frombuffer(mapping, dtype=torch.uint8) for mapping in mappings]
    for component in range(2):
        for page in (0, physical):
            raw[component][page * dim : (page + 1) * dim] = finite_codes[
                (torch.arange(dim) + component * 17 + page) % len(finite_codes)
            ]
    keys, values = [
        tensor.view(torch.float8_e4m3fn).reshape(pages, 1, 1, dim) for tensor in raw
    ]
    indices = torch.tensor([physical, 0, physical], dtype=index_dtype)
    output = torch.empty((3, 2, 1, 1, dim), dtype=torch.bfloat16)
    expected = torch.stack(
        [
            torch.stack([source[page].bfloat16() for source in (keys, values)])
            for page in (physical, 0, physical)
        ]
    )
    _expand_pages[(3, 1)](
        keys,
        values,
        indices,
        output,
        1,
        *keys.stride(),
        *values.stride(),
        1,
        1,
        dim,
        False,
        2048,
        num_warps=4,
    )
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    del keys, values, raw
    gc.collect()
    for mapping in mappings:
        mapping.close()
    return {
        "source_byte_offset": physical * dim,
        "index_dtype": str(index_dtype),
        "passed": True,
    }


rows = [
    check(*case)
    for case in itertools.product(
        ("HND", "NHD"),
        (1, 3, 32),
        ("split", "interleaved"),
        (torch.int32, torch.int64),
        (1, 2),
    )
]
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
large = [check_large_offset(dtype) for dtype in (torch.int32, torch.int64)]
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert after - before < 64 * 1024, "Large-offset check used unexpected physical memory"
assert not torch.cuda.is_initialized()
source = Path(fp8_prefill_cache.__file__)
report = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "toolchain": {"torch": torch.__version__, "triton": triton.__version__},
    "mode": "actual Triton source executed by CPU interpreter",
    "finite_e4m3_encodings": 254,
    "ordinary_cases": rows,
    "large_offset_cases": large,
    "large_offset_peak_rss_growth_kib": after - before,
    "cuda_initialized": False,
    "limits": "Does not execute compiled GPU instructions, allocation/capture "
    "paths, FlashInfer attention or current-main serving integration.",
}
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "interpreter_cases_passed": len(rows),
            "large_offset_cases_passed": len(large),
            "finite_e4m3_encodings": 254,
            "large_offset_peak_rss_growth_kib": after - before,
            "cuda_initialized": False,
        }
    )
)
