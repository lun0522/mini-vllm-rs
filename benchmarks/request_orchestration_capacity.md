# Request Orchestration Capacity

The [request-routing policy benchmark](request_routing_policies.md) found
that estimated-work routing completed its workloads fastest by assigning 11 of
12 requests to the substantially faster GPU. This follow-up asks when the CPU
becomes useful despite that speed difference.

The measured profile for this workload puts GPU prefill at 2,551 tok/s versus
73 tok/s on CPU, a roughly 35× advantage. Decode narrows the gap to 293 versus
45 tok/s, or about 6.5×. The benchmark therefore pre-populates the same prefix
cache on both backends, leaving only a few input tokens to prefill, and uses
128-token outputs so decode dominates. This gives the CPU its best plausible
chance to help once GPU work accumulates.

The benchmark holds the routing policy, request shape, cached prefix,
throughput profiles, scheduler, and arrival interval constant while increasing
the measured request count. It compares **cache-aware least estimated work**
with a forced-GPU control to determine when using the slower CPU adds useful
parallel capacity.

**Scope:**

- This is a controlled capacity experiment on one machine and one small model,
  not a sustained-arrival workload or a general CPU/GPU crossover threshold.
- The benchmark studies one routing policy using the measured throughput-profile
  matrix. It does not repeat the five-policy comparison from the earlier report.
- Prompt diversity, natural stopping behavior, output quality, speculative
  decoding, and alternative scheduler settings are out of scope.

## Reproducibility

| Item | Details |
|---|---|
| Hardware | Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified memory |
| Operating system | macOS 27.0.1 |
| Model | Qwen2.5 0.5B Instruct Q4_K_M, without a draft model |
| Revisions | [`mini-vllm-rs` `cacfaef`](https://github.com/lun0522/mini-vllm-rs/commit/cacfaef7109e8b7ea76d0fb8544a0f9d39cc0503) and [`mini-vllm-eval` `b7ce000`](https://github.com/lun0522/mini-vllm-eval/commit/b7ce000dd19f3e2233f7e05f25cdcbbb6cadc05a) |
| Procedure | Follow the `request-orchestration-capacity-benchmark` skill in the `mini-vllm-eval` repository and run `request_orchestration_capacity` |
| Configuration | Mixed CPU/Metal GPU inference with F32 activations; **cache-aware least estimated work** routing using the measured 12-point [throughput-profile matrix](backend_throughput_profiles.textproto); separate 256 MiB paged prefix caches; maximum 2,048 batched tokens and 4 active requests per backend |
| Requests | Approximately 516 input tokens, 128 output tokens, EOS ignored; submissions start 50 ms apart and responses are consumed concurrently |
| Measurements | 3 sequential untraced invocations; every case starts a fresh server; values report the mean and sample standard deviation |
| Environment overrides | Tracing disabled; `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` unset |

The 50 ms interval and single preprocessing thread make request order
repeatable while allowing earlier requests to remain in flight.

## Workloads

Backend 1 runs on the CPU, and backend 2 runs on the Metal GPU. Before each
measured workload, two unmeasured **seed requests** place the same approximately
512-token prefix in both backend caches. Every measured request then uses the
same 516-token prompt, restores the seeded prefix, and generates 128 tokens.

Each workload runs twice on a fresh server:

- **Forced GPU:** A benchmark-only request field assigns every measured request
  to backend 2.
- **Mixed routing:** **Cache-aware least estimated work** selects between the
  CPU and GPU from cache-affinity snapshots, published load, and outstanding
  ingress reservations for dispatched work not yet reflected in worker load.

Exploratory runs varied the request count and found that CPU offloading began
with the 8th request; 7 requests still stayed entirely on the GPU. The measured
crossover case therefore uses 8 requests to capture the first CPU assignment.

| Workload | Requests | Purpose |
|---|---:|---|
| `GPU only` | 4 | Fits one full GPU active set and checks that the policy avoids the slower CPU while the GPU has immediate capacity. |
| `Crossover` | 8 | Adds enough GPU backlog to capture the first calibrated CPU assignment. |
| `Saturated` | 48 | Maintains a long GPU backlog so useful parallel CPU capacity can emerge. |

The workload names describe this measured configuration. They should be
recalibrated after changing the model, hardware, request shape, pacing, or
throughput profiles.

## Results

### Metrics

- **Wall time:** Time from the first measured submission until all responses
  have been consumed; excludes server startup and seed requests.
- **Aggregate output throughput:** Total generated tokens divided by wall time.
- **TTFT (time to first token):** Time from client submission until its first
  streamed output.
- **E2E (end-to-end latency):** Time from client submission until its final
  streamed output.
- **p25, p75, and p95:** The 25th-, 75th-, and 95th-percentile request
  latencies within one benchmark invocation.
- **Restored tokens:** Page-aligned input tokens reused from a backend's prefix
  cache, summed across measured requests.

### CPU Use Increased with GPU Backlog

Mixed routing kept the 4-request workload entirely on the GPU. It consistently
sent request 8 to CPU in the crossover workload, then assigned 6 of the 48
saturated requests to CPU in every run:

| Workload | CPU / GPU requests | Closest estimated-score margin |
|---|---:|---:|
| `GPU only` | 0 / 4 | 1.369 ± 0.000 s |
| `Crossover` | 1 / 7 | 0.089 ± 0.002 s |
| `Saturated` | 6 / 42 | 0.008 ± 0.003 s |

The score margin is the absolute difference between the two candidates'
estimated completion durations. The first saturated run below shows each
request in arrival order: blue selected the GPU, orange selected the CPU, and
lighter shades indicate closer scores. Each block also gives its absolute score
margin. The first CPU assignment was repeatable and not an equal-score tie, but
later decisions approached the crossover as GPU reservations accumulated.

![Routing decisions and estimated-score margins for one saturated run](assets/request_orchestration_capacity/saturated_routing_decisions.svg)

Requests 8, 16, 31, 39, and 46 went to CPU in every run. The remaining CPU
assignment was request 23 in run 1 and request 24 in runs 2 and 3. The stable
CPU count and one-request shift are consistent with routing decisions close to
the estimated crossover; in run 1, request 38 stayed on GPU by only 0.006 s.

The crossover workload gives the corrected boundary directly. Request 7 stayed
on GPU by **0.089 ± 0.002 seconds**, while request 8 went to CPU by **0.328 ±
0.002 seconds**. These are estimated completion-time differences between the
selected backend and the alternative, not measured request latencies.

This is the intended heterogeneous placement: the policy did not try to split
requests evenly. It used the CPU occasionally when parallel execution there
was estimated to finish sooner than joining the GPU backlog.

### CPU Offload Improved Capacity

Wall time shows when mixed routing began benefiting from additional CPU
capacity relative to the forced-GPU control.

![Wall time for forced-GPU and mixed routing as request count increases](assets/request_orchestration_capacity/wall_time.svg)

- At 4 requests, mixed routing and the forced-GPU control had effectively equal
  wall time.
- At 8 requests, moving the final request to CPU reduced wall time from
  **11.474 to 10.348 seconds** and increased aggregate output throughput from
  **89.245 to 98.952 tok/s**.
- At 48 requests, assigning 6 requests to CPU reduced wall time from **68.426
  to 60.883 seconds** and increased throughput from **89.790 to 100.915 tok/s**.

Capacity and latency did not move uniformly. The chart reports mixed routing's
relative change from its paired forced-GPU control; whiskers show sample
standard deviation across the three runs.

![Capacity and latency changes under mixed routing](assets/request_orchestration_capacity/capacity_latency_changes.svg)

The 8-request case shows a capacity crossover, but not a uniform latency win.
CPU offload shortened wall time and p95 E2E latency by 9.8%. Median TTFT fell
by 93.1% and median E2E latency fell by 29.8%, while p95 TTFT increased by 3.5%.
The median TTFT change is especially sensitive to this small, even-sized
sample: moving one request to CPU placed its first token in the lower half of
the eight observations, shifting the median from between the two GPU waves to
within the earlier group.

### Offload Shifted the Saturated TTFT Distribution

The saturated benefit was distributed across the request set: mixed routing
shifted the TTFT distribution lower at every reported percentile.

![Saturated TTFT distribution for forced-GPU and mixed routing](assets/request_orchestration_capacity/saturated_ttft.svg)

The largest relative improvement occurred in the lower half of the
distribution because early CPU assignments began producing output while the
GPU queue grew. Later requests also benefited as those assignments removed work
from the GPU backlog. Median E2E latency fell by **20.8%**, and p95 E2E latency
fell by **15.7%**. These improvements and the six CPU assignments were stable
across all three runs.

## Conclusions

1. **The policy avoided unnecessary CPU work at low concurrency.** It routed all
   4-request workloads to the GPU; mixed-routing wall time stayed within 0.8%
   of the forced-GPU control.
2. **The eighth request crossed the measured routing boundary.** Request 7
   remained on GPU, while sending request 8 to CPU reduced wall time by 9.8%
   versus forcing all eight onto GPU.
3. **A slower backend still added useful capacity under sustained backlog.**
   Six CPU assignments in the 48-request workload reduced wall time by 11.0%
   and raised aggregate output throughput by 12.4%.
4. **Useful offload did not require an even split.** Routing one-eighth
   of saturated requests to CPU improved the TTFT and E2E distributions while
   leaving most work on the faster GPU.
5. **The crossover is workload-specific.** Equal prefix availability, fixed
   128-token outputs, paced submission, ingress reservations, and measured
   throughput profiles all influence when the CPU becomes competitive. Cached
   prefixes avoided expensive CPU prefill; uncached inputs would move the
   crossover.

## Appendix: Complete Measurements

### Case Results

Values are mean ± sample standard deviation across three invocations. Restored
tokens are summed over measured requests and were identical across runs.

| Workload | Routing | CPU / GPU | Wall (s) | Output (tok/s) | Median TTFT (ms) | p95 TTFT (ms) | Median E2E (s) | p95 E2E (s) | Restored tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GPU only` | Forced GPU | 0 / 4 | 5.809 ± 0.046 | 88.136 ± 0.689 | 54.376 ± 5.850 | 67.674 ± 10.821 | 5.672 ± 0.040 | 5.691 ± 0.034 | 1,984 |
| `GPU only` | Mixed | 0 / 4 | 5.856 ± 0.075 | 87.446 ± 1.119 | 54.223 ± 1.618 | 72.142 ± 14.372 | 5.724 ± 0.076 | 5.746 ± 0.078 | 1,984 |
| `Crossover` | Forced GPU | 0 / 8 | 11.474 ± 0.011 | 89.245 ± 0.082 | 2,788.929 ± 1.766 | 5,559.789 ± 5.432 | 8.407 ± 0.007 | 11.166 ± 0.011 | 4,032 |
| `Crossover` | Mixed | 1 / 7 | 10.348 ± 0.011 | 98.952 ± 0.103 | 193.503 ± 6.610 | 5,754.640 ± 11.204 | 5.904 ± 0.011 | 10.067 ± 0.009 | 4,016 |
| `Saturated` | Forced GPU | 0 / 48 | 68.426 ± 0.072 | 89.790 ± 0.094 | 30,306.568 ± 11.654 | 60,481.232 ± 45.174 | 35.947 ± 0.012 | 66.088 ± 0.079 | 24,512 |
| `Saturated` | Mixed | 6 / 42 | 60.883 ± 0.065 | 100.915 ± 0.108 | 22,852.661 ± 27.925 | 50,103.683 ± 49.946 | 28.480 ± 0.033 | 55.731 ± 0.056 | 24,448 |

The 64-token difference is 0.26% of the forced-GPU total. In the forced-GPU
case, 4 requests restored 496 tokens and 44 restored 512; mixed routing had 8
and 40 requests in those groups. This reflects backend-local, page-aligned
cache growth during the concurrent replay. Every measured request restored a
nonzero prefix.

### Saturated TTFT Distribution

The table reports the mean of each invocation's distribution statistic.

| Routing | Minimum | p25 | Median | p75 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Forced GPU | 0.039 s | 15.201 s | 30.307 s | 45.404 s | 60.467 s | 60.499 s |
| Mixed | 0.039 s | 10.112 s | 22.853 s | 39.182 s | 50.102 s | 55.516 s |

### Variance and Validation

The 4- and 8-request backend assignments were identical across all three runs.
The saturated case used CPU six times in every run. Five CPU assignments were
identical; the remaining assignment shifted from request 23 in run 1 to request
24 in runs 2 and 3, as expected from the narrow score margins. The analyzer also
verified both seed destinations, nonzero cache reuse, request IDs, routing
epochs, candidate duration estimates, and selection of the minimum estimated
score for every mixed-routing request.

All wall-time and aggregate-throughput coefficients of variation (CV) were at
most 1.29%. Three 4-request TTFT metrics exceeded 5%, with a maximum of 19.92%
for mixed-routing p95 TTFT. Every saturated headline metric had at most 0.13%
CV, including 0.10% for mixed-routing p95 TTFT.

The 4-request TTFT percentages are sensitive to millisecond-scale changes and
distribution statistics over only four requests. Placement, wall time, output
throughput, and saturated tail latency were stable. No runs were discarded.
