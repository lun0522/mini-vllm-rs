# Request Orchestration

This benchmark studies how request-routing policies distribute concurrent work
between one CPU backend and one GPU backend. The first experiment below exposed
a gap between routing and worker load publication, which motivated adding
orchestrator-side ingress reservations before collecting the final policy
comparison.

## Reproducibility

| Item | Details |
|---|---|
| Hardware | Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified memory |
| Operating system | macOS 27.0.1 |
| Model | Qwen2.5 0.5B Instruct Q4_K_M, without a draft model |
| Revisions | [`mini-vllm-rs` `2fe3c97`](https://github.com/lun0522/mini-vllm-rs/commit/2fe3c97805c0ab2378499ff607d5882ae5f28ebb) and [`mini-vllm-eval` `02fa62c`](https://github.com/lun0522/mini-vllm-eval/commit/02fa62c781cfcfd47dffc9adf3ac032034644576) |
| Procedure | Follow the `request-orchestration-benchmark` skill in the `mini-vllm-eval` repository |
| Configuration | Mixed CPU/GPU inference with F32 activations, separate 256 MiB paged prefix caches, a maximum batched token count of 2,048, and four active requests per backend |
| Workloads | A shared-prefix seed followed by ten concurrent requests: shared-prefix and unique-prompt variants of five input/output shapes from 128/16 to 2,048/1,024 tokens |
| Throughput profiles | The estimated-work policies used either the measured 12-point [backend throughput profiles](backend_throughput_profiles.md) or the fallback profile |
| Measurements | One untraced diagnostic run; results are a behavioral baseline, not a stable performance comparison |
| Thread settings | `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` unset |

## Pre-reservation Baseline

### Results

| Policy | Profiles | CPU / GPU requests | Wall time (s) | Output (tok/s) | Shared tokens restored |
|---|---|---:|---:|---:|---:|
| Round robin | Not used | 5 / 5 | 79.7 | 42.6 | 4,128 |
| Least requests | Not used | 5 / 5 | 120.6 | 28.1 | 192 |
| Least estimated work | Detailed | 0 / 10 | 77.5 | 43.7 | 4,704 |
| Most cached prefix | Not used | 6 / 4 | 64.4 | 52.7 | 4,704 |
| Cache-aware least estimated work | Detailed | 0 / 10 | 77.6 | 43.7 | 4,704 |
| Least estimated work | Fallback | 0 / 10 | 77.4 | 43.9 | 4,704 |
| Cache-aware least estimated work | Fallback | 1 / 9 | 75.6 | 44.9 | 4,704 |

The cache-aware policies restored every available shared-prefix token, while
unique prompts restored none. Prefix-affinity publication and lookup therefore
worked as intended in this workload.

The routing distribution revealed a more important problem. Both detailed
estimated-work policies sent all ten concurrent requests to the GPU. Their
routing decisions were made faster than the selected worker could admit the
requests and publish updated operational state, so several decisions observed
the same apparently idle backend. The fallback profiles behaved similarly;
only the cache-aware fallback sent one small request to the CPU. Consequently,
this run cannot distinguish the quality of the detailed and fallback profiles.

Request count alone was also insufficient for balancing heterogeneous work.
Although `least-requests` produced an even 5/5 request split, the CPU received
expensive decode work, including a 1,024-output-token request. Its wall time was
120.6 seconds, compared with 79.7 seconds for round robin. Equal request counts
did not imply equal backend work.

`most-cached-prefix` was fastest in this particular run. It retained all 4,704
shared-prefix tokens and distributed uncached requests across both devices.
That result should not be read as a final policy ranking: it is a single sample,
and the load-aware policies were operating with incomplete ingress accounting.

### Why Ingress Reservations Were Needed

Worker snapshots describe work the worker has already observed. They do not
include requests that the orchestrator has selected and dispatched but that the
worker has not yet admitted. During a burst, that gap lets successive routing
decisions repeatedly select the same backend from an unchanged snapshot.

The orchestrator therefore needs to reserve the selected request's expected
work as part of the routing decision. Subsequent decisions can combine the
latest worker snapshot with this provisional ingress work. The reservation is
released after the worker incorporates the request into its published state,
or rolled back if dispatch fails.

The complete policy benchmark should be repeated after reservations are added.
At that point, three runs will be collected for every policy and throughput
profile configuration, with two additional runs if a headline metric exceeds a
5% coefficient of variation. Those post-reservation measurements, rather than
this diagnostic run, will support the final policy comparison.
