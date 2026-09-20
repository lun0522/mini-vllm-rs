# CPU Paged Attention

We added a simple CPU paged-attention implementation and tested a few ways to
reduce its memory use and execution time. The benchmark uses the Qwen2.5 0.5B
model and a paged KV cache with 16 tokens per page. Each server handles three
sequential requests that generate 1, 512, and 2048 output tokens.

**Scope:**

CPU inference is not the main target of this project. We therefore focus on
simple optimizations rather than trying to build an optimal CPU paged-attention
kernel.

**Reproducibility:**

- Hardware: Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified
  memory.
- Operating system: macOS 26.6.2.
- Model: Qwen2.5 0.5B Instruct Q4_K_M, using a paged KV cache with 16 tokens per
  page.
- Revisions:
  [`mini-vllm-rs` `2bc67ec`](https://github.com/lun0522/mini-vllm-rs/commit/2bc67ec7d7043fb2256e7e9eb61b263f864eb05b)
  and
  [`mini-vllm-eval` `9b72cb3`](https://github.com/lun0522/mini-vllm-eval/commit/9b72cb3df02f05955a4c288a752cd9d8639c93be).
- Command, run from the `mini-vllm-eval` repository:

  ```shell
  python3 main.py --benchmark cpu_paged_attention
  ```

- Each configuration was measured once. `CANDLE_NUM_THREADS` and
  `RAYON_NUM_THREADS` were unset; the benchmark harness set only the
  configuration-specific CPU attention variables.

### Implementations

Candle does not support paged attention on CPU. Our baseline uses paged KV-cache
management but reconstructs contiguous K and V tensors before calculating
attention. In other words, the cache is paged, while the attention calculation
remains contiguous.

This benchmark compares three choices:

1. **Contiguous vs. paged attention:** Contiguous attention reconstructs full K
   and V before calculating attention. Paged attention calculates query-key
   scores one cache page at a time, then joins the score slices before softmax.
2. **Repeated KV vs. grouped Q:** For GQA, the shape of Q is
   `[1, num_q_heads, q_len, head_dim]`, while K and V have shape
   `[1, num_kv_heads, kv_len, head_dim]`. The baseline path copies each K and V
   head to match the number of query heads. The grouped-Q path instead reshapes
   Q to group query heads that share a KV head:
   `[num_kv_heads, repetition_count * q_len, head_dim]`. This allows each group
   to use its original K and V head without copying them.
3. **Upfront full V vs. concatenated V vs. page-wise V:**
   - **Upfront full V** reconstructs both full K and full V before contiguous
     attention, as in the baseline path.
   - **Concatenated V** produces the same V shape and values, but only after the
     paged query-key calculation and softmax. K remains page-wise.
   - **Page-wise V** never reconstructs full V. It multiplies each attention-
     weight slice by its matching V page, then adds the page outputs.

### Results

#### Output Token Count: 1

| Attention | Q/KV layout | V matmul | TTFT (ms) | E2E (s) | E2E vs baseline | Total (tok/s) | Decode (tok/s) | Decode vs baseline |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Contiguous | Repeated KV | Upfront full V | 1721.12 | 1.721 | 1.00x | 0.58 | - | - |
| Contiguous | Grouped Q | Upfront full V | 1689.37 | 1.689 | 1.02x | 0.59 | - | - |
| Paged | Repeated KV | Concatenated V | 1728.34 | 1.728 | 1.00x | 0.58 | - | - |
| Paged | Repeated KV | Page-wise V | 1762.07 | 1.762 | 0.98x | 0.57 | - | - |
| Paged | Grouped Q | Concatenated V | 1698.52 | 1.699 | 1.01x | 0.59 | - | - |
| Paged | Grouped Q | Page-wise V | 1713.40 | 1.713 | 1.00x | 0.58 | - | - |

#### Output Token Count: 512

| Attention | Q/KV layout | V matmul | TTFT (ms) | E2E (s) | E2E vs baseline | Total (tok/s) | Decode (tok/s) | Decode vs baseline |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Contiguous | Repeated KV | Upfront full V | 1708.33 | 13.663 | 1.00x | 37.47 | 42.74 | 1.00x |
| Contiguous | Grouped Q | Upfront full V | 1682.23 | 12.355 | 1.11x | 41.44 | 47.88 | 1.12x |
| Paged | Repeated KV | Concatenated V | 1703.08 | 14.243 | 0.96x | 35.95 | 40.75 | 0.95x |
| Paged | Repeated KV | Page-wise V | 1716.93 | 15.003 | 0.91x | 34.13 | 38.46 | 0.90x |
| Paged | Grouped Q | Concatenated V | 1699.60 | 12.762 | 1.07x | 40.12 | 46.19 | 1.08x |
| Paged | Grouped Q | Page-wise V | 1710.05 | 13.677 | 1.00x | 37.44 | 42.70 | 1.00x |

#### Output Token Count: 2048

| Attention | Q/KV layout | V matmul | TTFT (ms) | E2E (s) | E2E vs baseline | Total (tok/s) | Decode (tok/s) | Decode vs baseline |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Contiguous | Repeated KV | Upfront full V | 1709.63 | 68.953 | 1.00x | 29.70 | 30.44 | 1.00x |
| Contiguous | Grouped Q | Upfront full V | 1704.98 | 54.946 | 1.25x | 37.27 | 38.45 | 1.26x |
| Paged | Repeated KV | Concatenated V | 1710.67 | 75.488 | 0.91x | 27.13 | 27.75 | 0.91x |
| Paged | Repeated KV | Page-wise V | 1720.17 | 82.647 | 0.83x | 24.78 | 25.29 | 0.83x |
| Paged | Grouped Q | Concatenated V | 1695.58 | 59.284 | 1.16x | 34.55 | 35.55 | 1.17x |
| Paged | Grouped Q | Page-wise V | 1772.48 | 67.234 | 1.03x | 30.46 | 31.27 | 1.03x |

#### Peak RSS After the 1-Token Warm-up

| Attention | Q/KV layout | V matmul | Start RSS (MiB) | Peak RSS (MiB) | Peak increase (MiB) |
|---|---|---|---:|---:|---:|
| Contiguous | Repeated KV | Upfront full V | 1365.56 | 1506.05 | 140.48 |
| Contiguous | Grouped Q | Upfront full V | 1363.23 | 1378.09 | 14.86 |
| Paged | Repeated KV | Concatenated V | 1367.59 | 1489.48 | 121.89 |
| Paged | Repeated KV | Page-wise V | 1366.23 | 1381.89 | 15.66 |
| Paged | Grouped Q | Concatenated V | 1364.81 | 1381.09 | 16.28 |
| Paged | Grouped Q | Page-wise V | 1367.17 | 1383.19 | 16.02 |

### Analysis

Starting RSS stays within about 4.4 MiB across all six runs, so the peak
increases are comparable. The results match the tensor layouts used by each
implementation:

| Implementation | Memory behavior | Execution behavior |
|---|---|---|
| **Contiguous attention + repeated KV + upfront full V** (baseline) | Repeats full-length K and V across all query heads, producing the largest **140.48 MiB** increase. | Uses large matmuls, but copying full K and V becomes expensive as the cache grows. Decode falls to **30.44 tok/s** at 2048 tokens. |
| **Contiguous attention + grouped Q + upfront full V** | Keeps full-length K and V at the smaller KV-head count, producing the lowest **14.86 MiB** increase. | Avoids KV copies and keeps large, efficient matmuls. It is the fastest case at **38.45 tok/s**. |
| **Paged attention + repeated KV + concatenated V** | Repeats K one small page at a time, but still concatenates and repeats full-length V. Peak increase remains high at **121.89 MiB**. | Replaces the large query-key matmul with many small per-page matmuls and still copies full V, so it is slower than the contiguous baseline at **27.75 tok/s**. |
| **Paged attention + repeated KV + page-wise V** | Repeats only one K or V page at a time, keeping the peak increase low at **15.66 MiB**. | Replaces one large score-value matmul with many small ones and adds their results. This is the slowest case at **25.29 tok/s**. |
| **Paged attention + grouped Q + concatenated V** | Avoids repeating K and V. Concatenated V contains only the original KV heads, so peak increase stays low at **16.28 MiB**. | Keeps one large score-value matmul and is the fastest paged case at **35.55 tok/s**, though page operations still make it slower than contiguous grouped Q. |
| **Paged attention + grouped Q + page-wise V** | Avoids repetition and handles V one page at a time, producing a similar **16.02 MiB** increase. | Page-wise score-value matmuls add overhead without saving more memory, reducing decode to **31.27 tok/s**. |

The important memory cost comes from combining the full cache length with the
full query-head count. Grouped Q avoids the query-head expansion, while
page-wise V limits repetition to 16 tokens at a time. Either choice avoids the
large full-length repeated V tensor. Using both does not reduce RSS further in
this benchmark.

### Conclusions

1. Grouped Q is the useful optimization for both speed and memory. At 2048
   output tokens, contiguous grouped-Q attention is **1.26x** the baseline while
   reducing the peak RSS increase from **140.48 MiB** to **14.86 MiB**. It is
   therefore the default CPU implementation.
2. Paged grouped-Q attention with concatenated V is the best paged
   implementation. It keeps the low-memory behavior at **16.28 MiB** and reaches
   **1.17x** the baseline, but remains about 8% slower than contiguous grouped Q
   because it runs query-key attention page-by-page and joins the score slices.
3. Page-wise V is useful only when KV repetition is still enabled. It reduces
   the paged repeated-KV increase from **121.89 MiB** to **15.66 MiB**, but the
   many small matmuls make execution slower. With grouped Q already enabled, it
   provides no measurable memory improvement and remains slower than
   concatenated V.
4. If this project were to evolve into a production-grade CPU inference
   framework, the next step should be a fused attention kernel using online
   softmax, similar to FlashAttention. That would avoid building the full score
   tensor and the full V tensor. This is currently out of scope.
