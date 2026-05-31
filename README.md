# Dynamic KV-Cache for LLM Inference

A torch-native implementation of dynamic KV-Cache for autoregressive LLM inference, with torch.compile compatibility and benchmarking.

Built at GPT-2 scale: `d_model=256`, `n_heads=4`, `head_dim=64`.

---

## Project Structure

```
dynamic_kv_cache/
├── src/
│   ├── attention.py              
│   ├── static_kv_cache.py        
│   ├── dynamic_kv_cache.py       
│   ├── compiled_kv_cache.py      
│   └── paged_attention.py        
├── benchmark.py                  
├── results/
│   └── benchmark_results.csv
└── requirements.txt
```

---

## Phases

- **Phase 1** — Baseline causal attention with no caching
- **Phase 2** — Static KV-Cache (pre-allocated fixed tensors)
- **Phase 3** — Dynamic KV-Cache (growing via concatenation)
- **Phase 4** — torch.compile integration and graph break analysis
- **Phase 5** — PagedAttention concept (block-based memory management)
- **Phase 6** — Full benchmarks and trade-off analysis

---

## Benchmark Results

*To be filled in after Phase 6.*

| Variant | Mean Latency/step (ms) | Peak Memory (MB) | Notes |
|---|---|---|---|
| No Cache (baseline) | | | |
| Static KV-Cache | | | |
| Dynamic KV-Cache | | | |
| Dynamic + torch.compile | | | |

---

## Key Questions Answered

*To be filled in after Phase 6.*

- When would you use static vs dynamic KV-Cache in production?
- What does `torch.compile(dynamic=True)` cost vs gain?
- What problem does PagedAttention solve that neither cache type solves?
