# Continuous Batching

Continuous batching admits new requests while other requests are already
decoding. This should reduce the time that queued requests wait for their first
token and allow the inference backend to process more useful work together, but
sharing each model step may increase latency for requests already in progress.

We benchmarked both effects with Qwen2.5 7B and its 0.5B speculative draft
model. A 2-request handoff isolates the waiting request's time to first token
(TTFT), while a heterogeneous 4-request workload combines ongoing decode
with short and long prefills that cannot fit in one scheduling decision.

**Scope:**

- This benchmark validates continuous batching at modest local-inference
  concurrency. Production arrival distributions and high-concurrency serving
  are out of scope.
- All cases use first-come-first-served scheduling. Comparing scheduling
  policies belongs in a separate benchmark.
- Target-only execution, other model pairs, memory measurements, and tracing
  are also out of scope.

## Reproducibility

| Item | Details |
|---|---|
| Hardware | Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified memory |
| Operating system | macOS 26.6.2 |
| Models | Qwen2.5 7B Instruct Q4_K_M as the target and Qwen2.5 0.5B Instruct Q4_K_M as the fixed-4 speculative draft model |
| Revisions | [`mini-vllm-rs` `e99e7fb`](https://github.com/lun0522/mini-vllm-rs/commit/e99e7fb36454c0c64ff19d1ad0d076c527108b2f) and [`mini-vllm-eval` `f64a858`](https://github.com/lun0522/mini-vllm-eval/commit/f64a858ec9612a99559aee164c8dcb35652055ab) |
| Procedure | Follow the `continuous-batching-benchmark` skill in the `mini-vllm-eval` repository |
| Configuration | GPU inference with F32 activations, a paged KV cache with 16 tokens per page, a 1 GiB target KV cache, a maximum batched token count of 1024, first-come-first-served scheduling, and `MINI_VLLM_METAL_GEMV_MAX_ROWS=4` |
| Workloads | A 1-token warm-up in every fresh server, followed by either a 2-request handoff generating 512 tokens per request or a heterogeneous 4-request workload generating 256 tokens per request; EOS was ignored |
| Measurements | 3 untraced runs reporting the mean and sample standard deviation; the maximum coefficient of variation among follower mean TTFT, anchor E2E latency, and aggregate output throughput was 4.55%, so the workflow did not require 2 additional runs |
| Thread settings | `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` unset |

Generated text was not required to match across active request limits. The
2-request workload also produced different draft statistics across those
limits. A later [Metal GEMV/GEMM investigation](metal_gemv_numerical_differences.md)
found that batching changed the packed tensor shapes and selected a different
quantized matmul kernel. The resulting small logit differences changed a
near-tied greedy token choice, after which the output and draft acceptance
diverged.

## Workloads

Each fresh server first completes an unmeasured 1-token warm-up request. The
measured workload then starts with one anchor request. The followers are
submitted after the anchor produces its first streamed text, ensuring that
they arrive while it is decoding.

| Workload | Requests | Input tokens | Output tokens | Active request limits |
|---|---:|---|---|---|
| 2-request handoff | 2 | Anchor 128; follower 121 | 512 each | 1 and 2 |
| Heterogeneous 4-request | 4 | Anchor 123; followers 1068, 125, and 998 | 256 each | 1 and 4 |

With an active request limit of 1, the server queues followers and executes
requests serially. The higher limits admit every request in the corresponding
workload. In the heterogeneous case, the followers contain 2,191 input tokens,
more than 2× the 1,024-token scheduling budget. Their prefills therefore
span multiple scheduling decisions while the anchor continues decoding.

## Results

Wall time runs from anchor submission until every measured request completes.
Aggregate output throughput is total output tokens divided by wall time, while
case-level draft acceptance is total accepted draft tokens divided by total
proposed draft tokens across requests.

### 2-Request Handoff

Allowing both requests to remain active reduced follower TTFT from **63.227
seconds to 1.564 seconds**, while increasing aggregate output throughput from
**8.273 to 13.280 tokens/s**.

| Active request limit | Follower TTFT (s) | Anchor E2E (s) | Wall time (s) | Output (tok/s) | Draft acceptance |
|---:|---:|---:|---:|---:|---:|
| 1 | 63.227 ± 2.879 | 63.298 ± 2.873 | 123.909 ± 4.462 | 8.273 ± 0.300 | 32.9% |
| 2 | 1.564 ± 0.029 | 77.083 ± 1.549 | 77.127 ± 1.518 | 13.280 ± 0.262 | 37.1% |

Compared with serial admission, allowing both requests to remain active produced:

- Follower TTFT reduction: **97.5% ± 0.1%**
- Output-throughput speedup: **1.606× ± 0.042×**
- Anchor E2E change: **+21.9% ± 3.7%**

The follower began producing output almost immediately instead of waiting for
the anchor to finish. The anchor itself became 21.9% slower because it shared
decode work with the follower, but overlapping the requests reduced total wall
time by 37.8%.

#### Unexpected Draft-Acceptance Shift

Both cases used the same fixed proposal length of 4, but draft acceptance
increased from 32.9% to 37.1%: accepted/proposed totals changed from 619/1,883
to 664/1,788.

These totals were identical across all 3 runs, so the shift is specific to the
execution configuration rather than measurement noise. A later
[targeted investigation](metal_gemv_numerical_differences.md) found that target
verification diverged first: packed execution crossed the Metal GEMV threshold,
and small GEMV/GEMM numerical differences flipped a near-tied greedy token
choice. Per-verification totals matched the final statistics, and forcing both
cases through GEMV restored identical output and acceptance totals. The
4-request workload showed no comparable shift (41.2% to 41.9%), so higher
acceptance was not a general effect of continuous batching.

The shift cannot plausibly explain follower TTFT falling from 63.227 to 1.564
seconds, but it may contribute to throughput. Proposed draft tokens fell by
5.0%, from 1,883 to 1,788, while throughput rose by 60.6%. The 1.606× result is
therefore the observed end-to-end speedup, not an isolated batching-only
speedup.

### Heterogeneous 4-Request Workload

Continuous batching reduced mean follower TTFT from **72.013 to 22.545
seconds**. Maximum follower TTFT fell from **110.737 to 23.442 seconds**, while
aggregate throughput improved from **7.280 to 13.203 tokens/s**.

| Active request limit | Follower mean TTFT (s) | Follower max TTFT (s) | Anchor E2E (s) | Wall time (s) | Output (tok/s) | Draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 72.013 ± 1.355 | 110.737 ± 3.063 | 31.091 ± 0.599 | 140.716 ± 3.182 | 7.280 ± 0.166 | 41.2% |
| 4 | 22.545 ± 0.448 | 23.442 ± 0.463 | 77.551 ± 1.393 | 77.574 ± 1.401 | 13.203 ± 0.240 | 41.9% |

Compared with serial admission, allowing all 4 requests to remain active
produced:

- Follower mean TTFT reduction: **68.7% ± 0.9%**
- Output-throughput speedup: **1.814× ± 0.025×**
- Anchor E2E change: **+149.6% ± 9.3%**

The benefit to queued requests and total throughput came with a much larger
cost to the anchor: its E2E latency increased by 149.6%. All 4 continuously
batched requests completed at about 72–78 seconds, instead of the serial case's
staggered completions and 140.716-second wall time, but the original request no
longer completed early.

## Conclusions

1. For these fixed-4 speculative workloads, continuous batching validated its
   primary assumption: it substantially reduced follower TTFT and improved
   aggregate throughput.
2. The improvement is not free. Sharing execution increased anchor latency by
   21.9% with 1 follower and 149.6% with 3 mixed-prefill followers.
3. Active request capacity is therefore a latency-throughput tradeoff for local
   inference, not a setting that should be maximized without considering the
   desired responsiveness of requests already running.

## Appendix: Complete Measurements

The headline tables aggregate followers to compare queueing and overall
performance. The tables below retain every request role.

### 2-Request Measurements

| Active request limit | Role | Input | Output | TTFT (s) | E2E (s) | Draft acceptance |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Anchor (short) | 128 | 512 | 1.208 ± 0.018 | 63.298 ± 2.873 | 31.9% (305/957) |
| 1 | Follower (short) | 121 | 512 | 63.227 ± 2.879 | 122.644 ± 4.430 | 33.9% (314/926) |
| 2 | Anchor (short) | 128 | 512 | 1.223 ± 0.023 | 77.083 ± 1.549 | 33.9% (319/940) |
| 2 | Follower (short) | 121 | 512 | 1.564 ± 0.029 | 70.723 ± 1.459 | 40.7% (345/848) |

### Heterogeneous 4-Request Measurements

The continuously batched rows can be compared by request role. With an active
request limit of 1, however, the concurrently submitted followers do not
necessarily enter the FCFS queue in a deterministic order. Their individual
rows are therefore diagnostic rather than matched serial comparisons; the
case-level follower mean and maximum are the appropriate comparison.

| Active request limit | Role | Input | Output | TTFT (s) | E2E (s) | Draft acceptance |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Anchor (short) | 123 | 256 | 1.211 ± 0.071 | 31.091 ± 0.599 | 32.8% (157/479) |
| 1 | Follower 1 (long) | 1068 | 256 | 96.986 ± 20.766 | 126.118 ± 19.905 | 48.0% (189/394) |
| 1 | Follower 2 (short) | 125 | 256 | 43.920 ± 22.310 | 72.924 ± 21.249 | 37.8% (166/439) |
| 1 | Follower 3 (long) | 998 | 256 | 75.131 ± 37.078 | 103.218 ± 37.601 | 48.3% (189/391) |
| 4 | Anchor (short) | 123 | 256 | 1.186 ± 0.025 | 77.551 ± 1.393 | 33.5% (158/472) |
| 4 | Follower 1 (long) | 1068 | 256 | 23.123 ± 0.501 | 71.979 ± 1.024 | 48.0% (189/394) |
| 4 | Follower 2 (short) | 125 | 256 | 21.228 ± 0.429 | 74.460 ± 1.255 | 39.9% (171/429) |
| 4 | Follower 3 (long) | 998 | 256 | 23.285 ± 0.565 | 71.714 ± 1.420 | 48.3% (189/391) |
