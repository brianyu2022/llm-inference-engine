# Baseline — Stage 1 (NumPy)

- machine: arm64 / Darwin (NumPy 2.5.1, BLAS via the platform default)
- decode throughput: **10.0 tok/s** (32 tokens, prompt 16)
- per-token latency grew 88ms -> 110ms as context grew (the O(T^2) recompute we kill in Stage 3)

## Forward latency

| seq_len | ms/forward | tok/s (prefill) |
|--------:|-----------:|----------------:|
| 8 | 90.9 | 88.0 |
| 32 | 99.4 | 321.8 |
| 128 | 169.0 | 757.6 |
| 256 | 276.1 | 927.1 |
| 512 | 546.8 | 936.4 |
