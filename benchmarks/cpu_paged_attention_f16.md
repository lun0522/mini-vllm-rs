# CPU Paged Attention with F16 Activations

The [CPU activation-dtype benchmark](cpu_activation_dtype.md) fixes attention
at contiguous attention, grouped Q, and upfront full V so it can isolate the
effect of F16 activations. This benchmark instead fixes the activation dtype at
F16 and finds the best CPU attention implementation.

The earlier [CPU paged-attention benchmark](cpu_paged_attention.md) evaluated
the same implementation choices with F32 activations. F16 halves the size of
attention tensors and the KV cache, and its unquantized attention matmuls have
different performance characteristics, so the best F32 configuration is not
assumed to remain optimal.

**Scope:**

- Use F16 activations and KV caches with F16 quantized matmul routed through
  Candle's optimized F32 path.
- Compare contiguous and paged attention, repeated KV and grouped Q, and the
  available V-matmul strategies.
- Select the best default from the existing implementations. New fused or
  architecture-specific attention kernels are out of scope.

## Implementations

The benchmark compares the same six valid combinations as the original CPU
paged-attention benchmark:

| Attention | Q/KV layout | V matmul |
|---|---|---|
| Contiguous | Repeated KV | Upfront full V |
| Contiguous | Grouped Q | Upfront full V |
| Paged | Repeated KV | Concatenated V |
| Paged | Repeated KV | Page-wise V |
| Paged | Grouped Q | Concatenated V |
| Paged | Grouped Q | Page-wise V |

These choices isolate three decisions:

1. **Contiguous vs. paged attention:** Contiguous attention reconstructs full K
   and V tensors before attention. Paged attention computes query-key scores
   one cache page at a time.
2. **Repeated KV vs. grouped Q:** Repeated KV copies K and V heads to match the
   query-head count. Grouped Q reshapes queries so each group shares the
   original KV head without copying it.
3. **Upfront full V vs. concatenated V vs. page-wise V:** Upfront full V is
   available with contiguous attention. Paged attention either concatenates V
   before one large matmul or multiplies and accumulates V one page at a time.

## Measurements

Each implementation should handle the same warm-up, short-context, and
long-context requests. Compare:

- TTFT and end-to-end latency.
- Decode throughput as the context grows.
- Peak RSS increase after warm-up.
- Attention spans when tracing is needed to explain a result.

The preferred implementation should minimize memory without trading away a
material amount of decode throughput. Relative differences should be supported
by absolute measurements, especially when an operation is small.

## Results

### Small-Prefill

| Configuration | F32 TTFT (ms) | F16 TTFT (ms) | F16/F32 |
|---|---:|---:|---:|
| Contiguous / Repeated KV / Full V | 1596.880 ± 14.140 | 1622.642 ± 5.355 | 1.016× |
| Contiguous / Grouped Q / Full V | 1577.136 ± 4.098 | 1629.536 ± 65.782 | 1.033× |
| Paged / Repeated KV / Concatenated V | 1593.738 ± 7.256 | 1618.270 ± 6.163 | 1.015× |
| Paged / Repeated KV / Page-wise V | 1590.580 ± 9.020 | 1620.846 ± 27.626 | 1.019× |
| Paged / Grouped Q / Concatenated V | 1584.424 ± 4.179 | 1610.228 ± 3.574 | 1.016× |
| Paged / Grouped Q / Page-wise V | 1700.112 ± 194.104 | 1663.066 ± 59.365 | 0.978× |

### Long-Request

| Configuration | F32 TTFT (ms) | F16 TTFT (ms) | F16/F32 | F32 E2E (s) | F16 E2E (s) | F16/F32 | F32 decode (tok/s) | F16 decode (tok/s) | F16/F32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Contiguous / Repeated KV / Full V | 15612.840 ± 273.126 | 15817.540 ± 59.750 | 1.013× | 54.841 ± 0.479 | 48.497 ± 0.260 | 0.884× | 26.078 ± 0.220 | 31.304 ± 0.239 | 1.200× |
| Contiguous / Grouped Q / Full V | 15375.480 ± 96.875 | 15813.020 ± 114.870 | 1.028× | 45.090 ± 0.298 | 45.145 ± 0.164 | 1.001× | 34.428 ± 0.236 | 34.878 ± 0.143 | 1.013× |
| Paged / Repeated KV / Concatenated V | 16311.260 ± 111.217 | 16539.120 ± 32.957 | 1.014× | 60.165 ± 0.411 | 53.540 ± 0.184 | 0.890× | 23.328 ± 0.170 | 27.648 ± 0.132 | 1.185× |
| Paged / Repeated KV / Page-wise V | 17820.680 ± 52.770 | 17733.540 ± 72.512 | 0.995× | 66.512 ± 0.157 | 59.268 ± 0.208 | 0.891× | 21.012 ± 0.062 | 24.630 ± 0.111 | 1.172× |
| Paged / Grouped Q / Concatenated V | 15793.840 ± 58.915 | 16071.500 ± 32.239 | 1.018× | 48.464 ± 0.135 | 49.381 ± 0.152 | 1.019× | 31.314 ± 0.133 | 30.714 ± 0.141 | 0.981× |
| Paged / Grouped Q / Page-wise V | 16846.020 ± 426.761 | 16702.020 ± 87.313 | 0.991× | 54.966 ± 0.466 | 55.859 ± 0.308 | 1.016× | 26.838 ± 0.075 | 26.126 ± 0.202 | 0.973× |

### Long-Request Peak RSS Increase

| Configuration | F32 (MiB) | F16 (MiB) | F16/F32 |
|---|---:|---:|---:|
| Contiguous / Repeated KV / Full V | 262.352 ± 26.806 | 109.916 ± 36.242 | 0.419× |
| Contiguous / Grouped Q / Full V | 254.588 ± 3.980 | 70.736 ± 21.758 | 0.278× |
| Paged / Repeated KV / Concatenated V | 365.434 ± 39.503 | 72.522 ± 7.050 | 0.198× |
| Paged / Repeated KV / Page-wise V | 304.712 ± 3.527 | 75.054 ± 13.613 | 0.246× |
| Paged / Grouped Q / Concatenated V | 322.048 ± 16.053 | 77.782 ± 21.341 | 0.242× |
| Paged / Grouped Q / Page-wise V | 316.002 ± 5.360 | 76.698 ± 5.236 | 0.243× |
