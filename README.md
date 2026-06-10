# Dynamic KV-Cache for LLM Inference

A torch-native implementation of KV-caching strategies for autoregressive LLM inference, built from scratch as a learning project. Covers static allocation, dynamic growth, `torch.compile` integration, and block-based paged memory - benchmarked on both CPU and NVIDIA RTX 5070 Laptop GPU.

Benchmarked at: `d_model=512`, `n_heads=16`, `head_dim=64`, `batch=1`, `seq_len=512`, `steps=512`.

---

## Project Structure

```
dynamic_kv_cache/
├── src/
│   ├── attention.py           # Baseline causal multi-head attention + generate()
│   ├── static_kv_cache.py     # Pre-allocated fixed-size KV cache
│   ├── dynamic_kv_cache.py    # Dynamically growing KV cache via torch.cat
│   ├── compiled_kv_cache.py   # torch.compile(dynamic=True) integration
│   └── paged_attention.py     # Block-table-backed paged memory management
├── benchmark.py               # Full CPU + GPU benchmark runner
├── results/
│   └── benchmark_results.csv  # Raw benchmark output
└── requirements.txt
```

---

## What's Built

The project implements the full progression from naive autoregressive generation to production-grade memory management:

**Baseline** - No caching. Every decode step recomputes attention over the full growing sequence. Fast to write, but attention cost grows quadratically with sequence length.

**Static KV-Cache** - Pre-allocate tensors of shape `(batch, heads, max_seq_len, head_dim)` and write new keys/values into them with a position pointer. Decode cost is constant per step, but you must commit to a maximum sequence length at allocation time.

**Dynamic KV-Cache** - Start with `None` and grow by `torch.cat` along the sequence dimension each step. No `max_seq_len` required. Trades away static's memory certainty for flexibility with variable-length sequences.

**`torch.compile` with `dynamic=True`** - Compile the model (not the generate loop) to allow variable-length tensor shapes across steps. Provides meaningful GPU speedups but interacts poorly with stateful Python objects (like a position counter) inside the compiled region, triggering recompilation.

**PagedAttention** - A block-table-backed memory pool. A `BlockTable` manages a fixed set of `(block_size, head_dim)` blocks; a `PagedKVCache` maps sequence IDs to lists of physical block indices, allocating on demand and returning blocks to the free list when a sequence finishes. Solves per-sequence memory fragmentation for multi-tenant serving.

---

## Benchmark Results

Settings: `BATCH_SIZE=1`, `SEQ_LEN=512`, `D_MODEL=512`, `NUM_HEADS=16`, `MAX_SEQ_LEN=1024`, `STEPS=512`, `WARMUP_RUNS=10`

### CPU

| Variant | Prefill (ms) | Mean Decode (ms/step) | Total (ms) |
|---|---|---|---|
| No Cache | — | 20.68 | 10,589 |
| Compiled - No Cache | — | 17.32 | 8,866 |
| Static KV-Cache | 134.0 | 0.22 | 247 |
| Compiled - Static KV-Cache | 120.1 | 0.26 | 252 |
| Dynamic KV-Cache | 113.4 | 0.31 | 273 |
| Compiled - Dynamic KV-Cache | 137.6 | 0.36 | 322 |
| Paged KV-Cache | 16.7 | 2.08 | 1,081 |
| Compiled - Paged KV-Cache | 17.8 | 2.26 | 1,173 |

### GPU (RTX 5070 Laptop)

| Variant | Prefill (ms) | Mean Decode (ms/step) | Total (ms) |
|---|---|---|---|
| No Cache | — | 1.77 | 905 |
| Compiled - No Cache | — | 0.99 | 505 |
| Static KV-Cache | 246.0 | 0.21 | 356 |
| Compiled - Static KV-Cache | 107.8 | 0.22 | 221 |
| Dynamic KV-Cache | 92.1 | 0.19 | 190 |
| Compiled - Dynamic KV-Cache | 100.5 | 0.21 | 205 |
| Paged KV-Cache | 15.2 | 0.24 | 140 |
| Compiled - Paged KV-Cache | 14.6 | 0.25 | 144 |

---

## Key Questions Answered

### 1. When would you use static vs dynamic KV-Cache in production?

**Static** wins when you know your maximum sequence length up front and memory is not a constraint. It pre-allocates once (`max_seq_len` slots per layer) and each decode step is a simple in-place index write - no memory allocation at runtime. On CPU it achieves 0.22 ms/step, the fastest decode latency of any variant. This is the right choice for batch inference servers where you set a hard `max_seq_len` budget and fill sequences up to it.

**Dynamic** wins when sequence lengths are unpredictable or vary widely across requests. It starts with zero memory and grows by `torch.cat` only as needed - there is no waste from pre-allocating slots that never get used. On GPU, dynamic cache actually edges out static (0.19 ms vs 0.21 ms/step) because the smaller, exactly-sized K/V tensors are more bandwidth-efficient than slicing a larger pre-allocated block. The tradeoff is repeated allocation on every step, which hurts on CPU (0.31 ms/step vs 0.22 ms).

**Rule of thumb:** static for known-length, memory-rich production servers; dynamic for research, variable-length workloads, or when you cannot afford to pre-allocate.

---

### 2. What does `torch.compile(dynamic=True)` cost vs gain?

**The gain** is real but device-dependent. On GPU, compiled no-cache is **1.8× faster** (0.99 ms vs 1.77 ms/step) and compiled static cache cuts prefill time nearly in half (108 ms vs 246 ms). The Triton-backed kernel fusion eliminates Python dispatch overhead and fuses operations that would otherwise be separate CUDA launches.

**The cost** shows up in two ways:

1. **Compile time / warmup:** The first run (or first few runs per distinct shape) triggers JIT compilation. Without warmup runs the first step can be 10–100× slower. `dynamic=True` reduces the number of recompiles for variable shapes but does not eliminate them.

2. **Stateful Python objects break the graph.** The static cache's `pos` counter is a plain Python integer that increments every step. `torch.compile` sees a new value each step and recompiles. After 8 recompiles it hits `recompile_limit` and falls back to eager — which is why compiled static cache is *slightly slower* than uncompiled on CPU (0.26 ms vs 0.22 ms). Dynamic and paged caches show the same pattern. The fix in real systems is to move position tracking into tensors and avoid Python-side mutation inside the compiled region.

**Bottom line:** `torch.compile` is a net win on GPU for compute-heavy workloads (attention, projections). It provides marginal or negative benefit on CPU or for small models where the compilation overhead and graph break penalties outweigh the kernel fusion gains.

---

### 3. What problem does PagedAttention solve that neither static nor dynamic KV-Cache solves?

**Memory fragmentation and rigid per-sequence allocation.**

Static cache pre-allocates `max_seq_len` slots for every sequence in a batch, even if that sequence only uses 50 of them — the unused slots are wasted and cannot be lent to another sequence. Dynamic cache avoids the waste but still allocates one contiguous tensor per sequence; its memory cannot be shared or reshuffled between sequences mid-flight.

In a serving system with many concurrent requests of unpredictable length, both approaches waste GPU memory - and wasted memory means smaller batch sizes, which means lower throughput.

PagedAttention (the core idea behind vLLM) treats the KV store like virtual memory: a shared pool of fixed-size **blocks** (pages) is maintained, and each sequence is assigned only the blocks it currently needs. When a sequence finishes, its blocks go back into the free list for the next request. This means:

- **No pre-allocation waste** - sequences never hold memory they are not using.
- **Memory reuse across sequences** - finished requests immediately free blocks for new ones.
- **Enables continuous batching** - new requests can be inserted mid-generation without evicting others.

The benchmark confirms the tradeoff on a single sequence: paged decode is ~10× slower than static on CPU (2.08 ms vs 0.22 ms) because block indirection and concatenation add overhead per step. On GPU that gap nearly disappears (0.24 ms vs 0.21 ms) because the GPU hides the latency. Paged also has the **fastest total time on GPU** (140 ms) because its prefill is cheapest (15 ms) - it does not scan a large pre-allocated buffer.

The value of PagedAttention is not single-sequence latency; it is system-level throughput when serving hundreds of concurrent variable-length requests.

---

## Setup

```bash
pip install torch
pip install pandas
```

For `torch.compile` on Windows, an x64 MSVC toolchain is required (the Inductor backend calls `cl.exe`). Install Visual Studio with the C++ workload and activate the x64 Native Tools environment before running.

---

## Validation

All cache variants are validated with `torch.allclose(output_cached, output_no_cache, atol=1e-5)` against the baseline generate function in each module's `__main__` block.
