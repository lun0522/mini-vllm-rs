Following the implementation of continuous batching support, we collected
initial benchmark results. The test workload consists of **2 concurrent
requests** (with prompt lengths of **128** and **121** tokens, respectively),
generating **512 output tokens** per request.

**Notes on Scope:**

1. These runs are designed to verify correctness and detect early performance
   regressions after introducing the batched code path, rather than serve as a
   comprehensive benchmark suite.
2. The attention kernels are not yet fully optimized. Each request currently
   executes self-attention sequentially and independently.

### CPU Backend

| Case | TTFT (s) | E2E (s) | Draft accept | Wall (s) | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| Sequential Target-only | 17.51 / 9.47 | 112.89 / 112.67 | - | 112.93 | 9.07 |
| Batched Target-only | 9.67 / 17.33 | 110.89 / 111.01 | - | 111.04 | 9.22 |
| Sequential Speculative | 20.67 / 21.04 | 237.38 / 216.61 | 30.5% / 43.2% | 237.43 | 4.31 |
| Batched Speculative | 20.90 / 20.90 | 213.63 / 191.43 | 30.5% / 43.2% | 213.66 | 4.79 |

### GPU Backend

| Case | TTFT (s) | E2E (s) | Draft accept | Wall (s) | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| Sequential Target-only | 1.14 / 2.02 | 60.82 / 60.89 | - | 60.92 | 16.81 |
| Batched Target-only | 2.12 / 1.05 | 117.06 / 116.99 | - | 117.11 | 8.74 |
| Sequential Speculative | 2.74 / 2.47 | 127.46 / 119.70 | 32.2% / 40.7% | 127.50 | 8.03 |
| Batched Speculative | 2.59 / 2.59 | 91.58 / 84.09 | 32.2% / 40.7% | 91.61 | 11.18 |

### Key Observations

1. **CPU Execution:** Comparing "Sequential" vs. "Batched" shows roughly
   equivalent metrics. Batching provides minimal performance improvement.
2. **GPU Execution:** Performance diverges significantly based on decoding
   strategy:
   * **Speculative Decoding:** "Batched" outperforms "Sequential". This aligns
     with expectations, as speculative decoding increases the active batch size
     enough for GPU kernels to achieve better compute utilization.
   * **Target-Only Decoding:** "Batched" exhibits a severe performance
     regression compared to "Sequential", requiring further profiling to isolate
     the root cause.

### Root Cause Analysis

The regression stems from dispatch logic in Candle's Metal backend for quantized
matrix multiplication. During sequential decoding, single-request tensor shapes
are `[1, 1, hidden_dim]`. Candle detects `M = 1` and routes execution through
its highly optimized GEMV kernel:

```rust
if src_shape.dim(D::Minus2)? == 1 {
    return self.fwd_mv(self_shape, storage, layout);
}
```

In batched mode, packing two requests yields a shape of `[1, 2, hidden_dim]`.
This causes Candle to bypass GEMV and trigger the generic GEMM kernel
(`call_quantized_matmul_mm_t`). For very small batch sizes, running the general
GEMM kernel introduces significant overhead compared to executing separate GEMV
operations.

### Mitigation

To address this, we introduced the `MINI_VLLM_METAL_GEMV_MAX_ROWS` environment
variable. When active batch sizes fall below this threshold, the LHS matrix is
split into vectors to execute GEMV kernels. The results are then concatenated
before the next batched operation.

As a proof of concept, setting `MINI_VLLM_METAL_GEMV_MAX_ROWS=2` on the GPU
backend resolves the regression for this initial benchmark, bringing batched
throughput back on par with sequential execution:

| Case | TTFT (s) | E2E (s) | Draft accept | Wall (s) | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| Sequential Target-only | 1.12 / 2.08 | 63.23 / 63.36 | - | 63.39 | 16.15 |
| Batched Target-only | 2.17 / 1.08 | 63.00 / 62.93 | - | 63.03 | 16.25 |
| Sequential Speculative | 2.76 / 2.49 | 130.02 / 122.36 | 32.2% / 40.7% | 130.05 | 7.87 |
| Batched Speculative | 2.43 / 2.43 | 83.62 / 76.26 | 32.2% / 40.7% | 83.65 | 12.24 |

### Threshold Tuning

To determine the optimal threshold for disabling the GEMV fallback, we
benchmarked generation throughput (**Output tok/s**) with the "Target-only"
workload across increasing levels of concurrency:

| Case | 3 Concurrent Requests | 4 Concurrent Requests | 5 Concurrent Requests |
|---|---|---|---|
| Sequential | 17.41 | 17.22 | 17.44 |
| Batched | 13.05 | 16.55 | 19.92 |
| Sequential + GEMV Optimization | 17.09 | 17.45 | 17.44 |
| Batched + GEMV Optimization | 17.52 | 18.07 | 18.06 |

As shown above, **5 concurrent requests** is the crossover point where
"Batched + GEMV Optimization" begins to underperform standard "Batched". Beyond
this batch size, the cumulative overhead of dispatching multiple individual GEMV
kernels and concatenating their results outweighs the execution time of the
single generic GEMM kernel.

Based on these findings, we set the default value of
`MINI_VLLM_METAL_GEMV_MAX_ROWS` to **4**.
