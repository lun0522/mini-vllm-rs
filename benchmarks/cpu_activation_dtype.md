# CPU Activation Dtype

Local inference is often constrained by memory capacity and bandwidth. Using
F16 instead of F32 for activations and the KV cache halves their element size,
which allows longer contexts or more concurrent requests within the same
memory budget. F16 can also reduce the cost of bandwidth-bound computations
and may make matmul faster when the backend has an optimized
F16 kernel.

We added configurable F32 and F16 activation types, then benchmarked CPU
inference to determine whether F16 provides those benefits in practice. The
initial implementation reduced memory use and accelerated the unquantized
attention matmuls, but made the model substantially slower overall. Tracing
identified Candle's F16-input quantized matmul as the bottleneck. We therefore
tested a small workaround that converts F16 inputs to F32, uses Candle's
optimized F32 quantized matmul, and converts the result back to F16.

**Scope:**

The goal is not to implement a new quantized CPU kernel. We want a simple path
that preserves most of F16's memory savings while keeping performance close to
the existing F32 implementation.

**Reproducibility:**

- Hardware: Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified
  memory.
- Operating system: macOS 26.6.2.
- Model: Qwen2.5 0.5B Instruct Q4_K_M, using a paged KV cache with 16 tokens per
  page.
- Revisions: `mini-vllm-rs` based on `47a5bc2` and `mini-vllm-eval` based on
  `6355b85`, including the activation-dtype benchmark changes in their working
  trees.
- Attention implementation: contiguous attention, grouped Q, and upfront full
  V. `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` were unset.
- Each server handled a 1-token warm-up, a 121-input-token request generating
  1 token, and a 1068-input-token request generating 1024 tokens. The maximum
  batched token count was 1024.
- Overall latency, throughput, and RSS are the mean and sample standard
  deviation from five runs without tracing.

## Initial F16 Implementation

The first implementation uses F16 for runtime activations, dequantized token
embeddings and normalization weights, model constants, and the KV cache. Model
weights remain in their original GGUF quantized format.

The benchmark gives F32 and F16 the same KV-cache token capacity. This requires
128 MiB for F32 and 64 MiB for F16. During the long request, raw F16 reduced the
peak RSS increase from **266.03 MiB** to **107.73 MiB**, a **59.5%** reduction.

F16 was nevertheless substantially slower in every overall performance
measurement:

| Metric | F32 | F16 | F16 / F32 |
|---|---:|---:|---:|
| Small-prefill TTFT | 1601.78 ± 2.75 ms | 5526.66 ± 2.33 ms | 3.45x |
| Large-prefill TTFT | 15363.52 ± 27.97 ms | 50293.72 ± 93.13 ms | 3.27x |
| End-to-end latency | 45.009 ± 0.127 s | 112.532 ± 0.178 s | 2.50x |
| Decode throughput | 34.51 ± 0.17 tok/s | 16.44 ± 0.03 tok/s | 0.48x |
| Peak RSS increase | 266.03 ± 4.22 MiB | 107.73 ± 1.18 MiB | 0.40x |

Starting and absolute peak RSS varied substantially between processes, while
the peak increase during the measured long request was stable. We therefore use
the peak RSS increase for the memory comparison.

### Attention Was Faster With F16

The trace shows that the large-prefill query-key and attention-value matmuls
benefit from F16. These operations multiply unquantized tensors, so both
operands use the activation dtype. KV-cache access and append operations are
also faster with the smaller tensors.

| Large-prefill attention span | F32 | F16 | F16 / F32 |
|---|---:|---:|---:|
| Q × K | 179.18 ± 2.35 ms | 85.64 ± 0.51 ms | 0.48x |
| Attention × V | 195.58 ± 2.83 ms | 103.24 ± 1.35 ms | 0.53x |
| KV-cache access | 1.74 ± 0.05 ms | 1.26 ± 0.02 ms | 0.73x |
| KV-cache append | 4.88 ± 0.19 ms | 3.16 ± 0.12 ms | 0.65x |
| Mask and softmax | 1218.50 ± 25.70 ms | 1391.94 ± 4.29 ms | 1.14x |

The faster attention matmuls were too small to offset the regression elsewhere.
The traced `qmatmul` spans cover multiplication of activations by quantized
model weights. Comparing the same named large-prefill spans shows that each
major quantized matmul was about three to four times slower with F16 inputs:

| Large-prefill `qmatmul` operation | F32 | F16 | F16 / F32 |
|---|---:|---:|---:|
| Attention key | 160.40 ± 0.23 ms | 488.50 ± 0.24 ms | 3.05x |
| Attention query | 962.24 ± 13.50 ms | 3296.90 ± 28.67 ms | 3.43x |
| Attention value | 102.48 ± 5.71 ms | 298.32 ± 0.30 ms | 2.91x |
| Attention output | 946.89 ± 19.29 ms | 3300.86 ± 28.43 ms | 3.49x |
| MLP gate | 4934.84 ± 32.15 ms | 17747.42 ± 54.00 ms | 3.60x |
| MLP up | 4920.78 ± 57.42 ms | 17754.40 ± 31.24 ms | 3.61x |
| MLP down | 882.29 ± 20.87 ms | 3328.60 ± 3.02 ms | 3.77x |

The MLP gate and up projections contribute the most absolute time, which
explains why the model is substantially slower even though Q × K and
Attention × V are nearly twice as fast.

## F16 Quantized Matmul Via F32

We checked Candle 0.11.0's source code, specifically `quantized/mod.rs` and
`quantized/k_quants.rs`, and found that its CPU quantized-matmul dispatcher
treats F32 and F16 inputs differently. In simplified form, the F32 branch
attempts the specialized AArch64 Q4_K × Q8 kernel before falling back to the
generic implementation, while the F16 branch goes directly to
`matmul_t_f16`:

```rust
match input.dtype() {
    DType::F32 if weights_are_q4k && n.is_multiple_of(8) => {
        matmul_q4k_x8(/* ... */)
    }
    DType::F32 => weights.matmul_t(/* ... */),
    DType::F16 => weights.matmul_t_f16(/* ... */),
}
```

The specialized F32 path repacks Q4_K weights in eight-column groups, converts
each F32 activation row to Q8_K once, and evaluates eight output columns at a
time with an AArch64 dot-product kernel. The generic F16 implementation instead
converts an activation row element-by-element to a temporary F32 vector,
quantizes it, then loops over output columns and calls the single-column
`vec_dot` implementation:

```rust
let lhs_f32 = lhs_f16.iter().map(|x| x.to_f32()).collect();
VecDotType::from_float(&lhs_f32, lhs_quantized);
for output_column in output_columns {
    output_column = f16::from_f32(vec_dot(/* ... */));
}
```

This explains why F16 does not benefit from the optimized eight-column kernel
on this CPU. Implementing and maintaining another architecture-specific
quantized kernel in this project would be a much larger change.

The workaround converts only the input activation from F16 to F32, invokes
Candle's existing quantized matmul, and converts the output back to F16. The
weights remain quantized throughout; it does not retain an F32 or F16 copy of
the weight matrices. The activation conversion is small compared with the matmul
and lets us reuse Candle's optimized kernel.

### Overall Results

The "F16-QMatMul" column corresponds to the matmul via F32 workaround:

| Metric | F32 | F16 | F16-QMatMul |
|---|---:|---:|---:|
| Small-prefill TTFT | 1601.78 ± 2.75 ms | 5526.66 ± 2.33 ms | 1649.62 ± 52.08 ms |
| Large-prefill TTFT | 15363.52 ± 27.97 ms | 50293.72 ± 93.13 ms | 15746.92 ± 83.21 ms |
| End-to-end latency | 45.009 ± 0.127 s | 112.532 ± 0.178 s | 45.210 ± 0.530 s |
| Decode throughput | 34.51 ± 0.17 tok/s | 16.44 ± 0.03 tok/s | 34.73 ± 0.53 tok/s |
| Peak RSS increase | 266.03 ± 4.22 MiB | 107.73 ± 1.18 MiB | 121.54 ± 0.65 MiB |

Compared with F32, F16-QMatMul is:

- **2.5%** slower for large-prefill TTFT.
- **0.4%** slower end-to-end.
- Effectively tied for decode throughput.
- **54.3%** lower in peak RSS increase.
- **3.0%** slower on average for small-prefill TTFT. One 1742.72 ms outlier
  increased both the mean and standard deviation; the other four measurements
  were between 1624.22 and 1629.00 ms, about 1–2% slower than F32.

### Quantized Matmul Trace

The traced aggregate confirms that the workaround fixes the original
bottleneck in the same named large-prefill `qmatmul` spans:

| Large-prefill `qmatmul` operation | F32 | F16 | F16-QMatMul | F16-QMatMul / F32 |
|---|---:|---:|---:|---:|
| Attention key | 160.40 ± 0.23 ms | 488.50 ± 0.24 ms | 184.39 ± 29.20 ms | 1.15x |
| Attention query | 962.24 ± 13.50 ms | 3296.90 ± 28.67 ms | 1001.34 ± 54.81 ms | 1.04x |
| Attention value | 102.48 ± 5.71 ms | 298.32 ± 0.30 ms | 111.88 ± 5.81 ms | 1.09x |
| Attention output | 946.89 ± 19.29 ms | 3300.86 ± 28.43 ms | 983.59 ± 53.65 ms | 1.04x |
| MLP gate | 4934.84 ± 32.15 ms | 17747.42 ± 54.00 ms | 5020.18 ± 174.89 ms | 1.02x |
| MLP up | 4920.78 ± 57.42 ms | 17754.40 ± 31.24 ms | 5032.51 ± 209.36 ms | 1.02x |
| MLP down | 882.29 ± 20.87 ms | 3328.60 ± 3.02 ms | 951.84 ± 49.25 ms | 1.08x |

The optimized spans are generally within a few percent of F32. Smaller
operations show more relative overhead because activation conversion is a
larger fraction of their execution time, but their absolute contribution is
small.

The workaround applies only to quantized matmul. Attention spans that continue
to operate directly on F16 activations remain close to the native F16 path in
the short prefill, large prefill, and final long-context decode:

| Phase | Attention span | F32 | F16 | F16-QMatMul | F16-QMatMul / F16 |
|---|---|---:|---:|---:|---:|
| Short prefill | Q × K | 6.66 ± 0.39 ms | 5.61 ± 0.09 ms | 5.61 ± 0.05 ms | 1.00x |
| Short prefill | Attention × V | 6.14 ± 0.14 ms | 5.11 ± 0.32 ms | 5.06 ± 0.08 ms | 0.99x |
| Short prefill | KV-cache access | 0.342 ± 0.024 ms | 0.223 ± 0.010 ms | 0.228 ± 0.004 ms | 1.02x |
| Short prefill | KV-cache append | 0.713 ± 0.071 ms | 0.447 ± 0.013 ms | 0.532 ± 0.027 ms | 1.19x |
| Short prefill | Mask and softmax | 19.41 ± 0.13 ms | 22.58 ± 0.27 ms | 23.33 ± 0.15 ms | 1.03x |
| Large prefill | Q × K | 179.18 ± 2.35 ms | 85.64 ± 0.51 ms | 85.73 ± 0.44 ms | 1.00x |
| Large prefill | Attention × V | 195.58 ± 2.83 ms | 103.24 ± 1.35 ms | 105.24 ± 4.96 ms | 1.02x |
| Large prefill | KV-cache access | 1.738 ± 0.053 ms | 1.263 ± 0.022 ms | 1.116 ± 0.028 ms | 0.88x |
| Large prefill | KV-cache append | 4.881 ± 0.188 ms | 3.161 ± 0.119 ms | 3.219 ± 0.013 ms | 1.02x |
| Large prefill | Mask and softmax | 1218.50 ± 25.70 ms | 1391.94 ± 4.29 ms | 1467.90 ± 12.88 ms | 1.05x |
| Final decode | Q × K | 4.12 ± 0.17 ms | 4.28 ± 0.46 ms | 4.01 ± 0.09 ms | 0.94x |
| Final decode | Attention × V | 1.61 ± 0.01 ms | 1.35 ± 0.01 ms | 1.36 ± 0.02 ms | 1.01x |
| Final decode | KV-cache access | 4.190 ± 0.102 ms | 2.966 ± 0.063 ms | 3.040 ± 0.074 ms | 1.03x |
| Final decode | KV-cache append | 0.042 ± 0.005 ms | 0.034 ± 0.002 ms | 0.038 ± 0.004 ms | 1.12x |
| Final decode | Mask and softmax | 3.88 ± 0.03 ms | 4.62 ± 0.16 ms | 4.69 ± 0.10 ms | 1.02x |

Q × K and Attention × V remain effectively unchanged from native F16 in all
three phases. Most cache and mask differences are also small. Short-prefill and
final-decode KV-cache append are 19% and 12% slower respectively, but the
absolute increases are only 0.09 ms and 0.004 ms. Large-prefill mask and
softmax has the largest absolute regression at 5.5%, or 75.96 ms. That span is
not routed through the workaround, and the increase is about 0.5% of the full
large-prefill forward, so it does not materially change the overall result.

## Conclusions

1. Simply changing CPU activations to F16 is not sufficient. It reduces the
   measured peak RSS increase by **59.5%** and makes large attention matmuls
   roughly twice as fast, but Candle's native F16 quantized matmul makes small
   prefill **3.45x** slower, large prefill **3.27x** slower, and end-to-end
   generation **2.50x** slower.
2. Routing F16 quantized matmul through Candle's optimized F32 path recovers
   nearly all F32 performance without implementing a new kernel. End-to-end
   latency is within **0.4%**, and decode throughput is effectively unchanged.
3. The workaround retains most of the memory benefit because model weights stay
   quantized and the persistent activations and KV cache remain F16. Its peak
   RSS increase is **121.54 MiB**, **54.3%** lower than F32.
4. A native optimized Q4 × F16 kernel could remove the remaining activation
   conversions, but the current performance difference is small enough that
   maintaining a custom kernel is not justified by this benchmark.
