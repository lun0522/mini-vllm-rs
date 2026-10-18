# CPU and GPU Backend Throughput Profiles

This benchmark measures the CPU and GPU costs that can be used by the
`least-estimated-work` and `cache-aware-least-estimated-work` request-routing
policies. Prefill and decode throughput differ substantially, especially on the
GPU, so the routing policies estimate their durations independently:

```text
prefill throughput = input token count / prefill time
decode throughput = (output token count - 1) / decode time
estimated time = remaining prefill tokens / prefill throughput
               + remaining decode tokens / decode throughput
```

The first output token is sampled at the end of prefill, so a request producing
`N` output tokens performs `N - 1` timed decode steps.

## Environment

| Item | Details |
|---|---|
| Hardware | Mac mini with Apple M2, 8-core CPU, 10-core GPU, and 16 GB unified memory |
| Operating system | macOS 27.0.1 (26A434) |
| Model | Qwen2.5 0.5B Instruct Q4_K_M, without a draft model |
| Activation dtype | F32 |
| KV cache | Paged, 16 tokens per page, 256 MiB; prefix caching disabled |
| Scheduler | One active request, maximum batched token count of 2,048 |
| Measurement | One warm-up followed by three measured requests for every workload and backend; EOS ignored |
| Revisions | [`mini-vllm-rs` `196b34f`](https://github.com/lun0522/mini-vllm-rs/commit/196b34f2ab09f99cf9506fb7ca51b6ff63edbeea) and [`mini-vllm-eval` `4b1cc1a`](https://github.com/lun0522/mini-vllm-eval/commit/4b1cc1a533de357b3e232ed65d7d3fe5db399973) |

The matrix covers input lengths of 128, 512, and 2,048 tokens and maximum
output lengths of 16, 128, 512, and 1,024 tokens. CPU and GPU were measured
separately so their workloads did not compete for unified-memory bandwidth.

Three complete benchmark invocations were aggregated. The separate phase rates
and phase-duration coefficients of variation below use those invocations. Two
additional targeted invocations measured only the combined execution time of
the noisy GPU 128-input/16-output shape, so they cannot contribute to the
phase-level analysis.

## Results

| Input | Max output | CPU prefill (ms) | CPU prefill tok/s | CPU decode (ms) | CPU decode tok/s | GPU prefill (ms) | GPU prefill tok/s | GPU decode (ms) | GPU decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 1,707.585 | 75.0 | 287.125 | 52.2 | 10.557 | 12,124.3 | 33.948 | 441.8 |
| 128 | 128 | 1,707.631 | 75.0 | 2,579.691 | 49.2 | 10.076 | 12,703.5 | 332.100 | 382.4 |
| 128 | 512 | 1,708.847 | 74.9 | 10,632.925 | 48.1 | 10.591 | 12,085.7 | 1,291.426 | 395.7 |
| 128 | 1,024 | 1,708.955 | 74.9 | 23,153.493 | 44.2 | 9.117 | 14,039.7 | 3,410.575 | 299.9 |
| 512 | 16 | 6,986.286 | 73.3 | 327.970 | 45.7 | 201.623 | 2,539.4 | 74.092 | 202.5 |
| 512 | 128 | 6,980.933 | 73.3 | 2,825.699 | 44.9 | 200.725 | 2,550.8 | 433.418 | 293.0 |
| 512 | 512 | 6,970.756 | 73.4 | 12,028.555 | 42.5 | 199.839 | 2,562.1 | 1,986.325 | 257.3 |
| 512 | 1,024 | 6,943.983 | 73.7 | 25,811.554 | 39.6 | 199.494 | 2,566.5 | 4,773.149 | 214.3 |
| 2,048 | 16 | 32,159.051 | 63.7 | 468.325 | 32.0 | 4,259.175 | 480.8 | 161.360 | 93.0 |
| 2,048 | 128 | 32,126.531 | 63.7 | 3,989.058 | 31.8 | 4,256.686 | 481.1 | 1,390.546 | 91.3 |
| 2,048 | 512 | 32,147.854 | 63.7 | 16,609.048 | 30.8 | 4,255.779 | 481.2 | 6,340.362 | 80.6 |
| 2,048 | 1,024 | 32,137.609 | 63.7 | 34,442.078 | 29.7 | 4,266.051 | 480.1 | 14,672.705 | 69.7 |

## Stability

| Input | Max output | CPU prefill CV | CPU decode CV | GPU prefill CV | GPU decode CV |
|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 1.02% | 0.85% | 4.58% | 10.18% |
| 128 | 128 | 0.60% | 6.66% | 6.32% | 2.17% |
| 128 | 512 | 0.79% | 0.62% | 6.53% | 3.76% |
| 128 | 1,024 | 0.25% | 0.32% | 2.72% | 0.51% |
| 512 | 16 | 0.43% | 0.32% | 0.14% | 1.12% |
| 512 | 128 | 0.34% | 0.16% | 0.15% | 0.45% |
| 512 | 512 | 0.50% | 0.16% | 0.52% | 1.00% |
| 512 | 1,024 | 0.06% | 0.24% | 0.19% | 0.51% |
| 2,048 | 16 | 0.29% | 0.21% | 0.11% | 1.11% |
| 2,048 | 128 | 0.16% | 0.44% | 0.13% | 0.24% |
| 2,048 | 512 | 0.03% | 0.27% | 0.28% | 0.08% |
| 2,048 | 1,024 | 0.06% | 0.32% | 0.07% | 0.11% |

Four phase measurements exceed the 5% coefficient-of-variation threshold:
GPU decode for 128-input/16-output (10.18%), CPU decode and GPU prefill for
128-input/128-output (6.66% and 6.32%), and GPU prefill for
128-input/512-output (6.53%). These short phases are sensitive to fixed timing
overhead and run-to-run noise. Every phase for the 512- and 2,048-token input
workloads has a coefficient of variation no greater than 1.12%, so those
profiles are stable. The 128-token profiles remain useful as approximate
calibration points, but should not be treated as equally precise.

CPU prefill throughput is relatively stable across the 128- and 512-token
inputs, while GPU prefill throughput drops sharply for the 2,048-token input.
Decode throughput decreases as context grows on both devices. A routing model
should therefore retain workload-specific profiles rather than use one global
prefill rate and one global decode rate.

## Recommended routing profiles

The complete measured matrix, rounded to the integer tok/s values accepted by
the configuration, is stored in
[`backend_throughput_profiles.textproto`](backend_throughput_profiles.textproto).
Use it with `least-estimated-work` or `cache-aware-least-estimated-work`:

```shell
--request-orchestrator-config \
  'routing_policy: REQUEST_ROUTING_POLICY_LEAST_ESTIMATED_WORK' \
--backend-throughput-profiles-file \
  benchmarks/backend_throughput_profiles.textproto
```

The profile file intentionally contains only benchmark measurements, leaving
the routing policy in the ordinary request-orchestrator configuration. The
orchestrator selects the profile with the smallest Manhattan distance from the
request's input-token and maximum-output-token counts. Retaining all twelve
points captures the observed changes in prefill and decode throughput.

When no profile file or inline profiles are supplied, mini-vLLM uses only the
rounded 128-input/512-maximum-output row: CPU prefill/decode rates of `75`/`48`
tok/s and GPU rates of `12086`/`396` tok/s. That fallback is convenient for
smoke tests, but the complete matrix is recommended for routing benchmarks on
this machine.

These values are specific to this machine, model, dtype, and inference
implementation. They should be remeasured after changing any of those inputs.
