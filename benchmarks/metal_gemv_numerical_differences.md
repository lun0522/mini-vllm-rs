# Metal GEMV Threshold V2.0: Numerical Differences

The original [Metal GEMV threshold benchmark](metal_gemv_threshold.md) chose
when small quantized matrix multiplications should be split into row-wise GEMV
operations. That work focused on performance. This follow-up investigates a
different consequence of the same threshold: GEMV and quantized GEMM can
produce slightly different logits, and greedy decoding can amplify a small
difference into different text.

## Introduction

We first sent the same creative-writing request to Qwen2.5 7B twice. Both runs
used greedy decoding and a fresh server. With
`MINI_VLLM_METAL_GEMV_MAX_ROWS=4`, the 845-token prefill used quantized GEMM;
with a threshold of 1024, it was split into row-wise GEMV operations. The first
597 decoded characters matched exactly, including this passage:

> The greenhouse was a relic of Selene Nine’s failed agricultural experiments,
> its central aisle now a silent testament to past ambition and present decay.

The continuations then separated at the same narrative point:

| Prefill path | First different continuation |
|---|---|
| Quantized GEMM | “Mara moved to the second door, where an emergency terminal stood…” |
| Row-wise GEMV | “Mara’s flashlight beam cut through the gloom, illuminating rows of dry hydroponic beds…” |

Both 512-token responses remained about Mara and Ilyan investigating an
abandoned lunar greenhouse. They used similar prose and revisited the same
objects, technical constraints, and disagreement, but differed in wording,
paragraph structure, dialogue, and event order. Their character-level sequence
similarity was 41.8%, and their word-level sequence similarity was 34.5%.
Those numbers describe textual alignment, not semantic quality: once an early
token changes, otherwise similar ideas can appear at different positions.

Neither response is obviously wrong. The important observation is that greedy
decoding is deterministic for a fixed numerical execution path, but does not
guarantee identical output across mathematically equivalent kernels with
different floating-point accumulation behavior.

## Scope

- This is a numerical diagnosis, not a performance benchmark or output-quality
  evaluation.
- It does not determine whether GEMV or GEMM is more accurate. That would
  require comparison with a higher-precision reference.
- The investigation covers the Metal backend, Q4_K_M models, and the observed
  request shapes. Other backends, models, and quantization formats are out of
  scope.

## Reproducibility

| Item | Details |
|---|---|
| Hardware | Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified memory |
| Operating system | macOS 26.6.2 |
| Models | Qwen2.5 7B Instruct Q4_K_M as the target and, where noted, Qwen2.5 0.5B Instruct Q4_K_M as the fixed-4 speculative draft model |
| Revisions | [`mini-vllm-rs` `8dbe147`](https://github.com/lun0522/mini-vllm-rs/commit/8dbe14760ac2fbca8549dde8111082e4994a4c9b) and [`mini-vllm-eval` `f64a858`](https://github.com/lun0522/mini-vllm-eval/commit/f64a858ec9612a99559aee164c8dcb35652055ab) |
| Initial comparison | One target-only creative-writing request with 845 input tokens and exactly 512 output tokens; fresh servers used thresholds 4 and 1024 |
| Main diagnostic | The continuous-batching benchmark's 2-request handoff, reduced to one run per active request limit and at most 256 output tokens per request |
| Configuration | GPU inference with F32 activations, Q4_K_M weights, a paged KV cache with 16 tokens per page, fixed-4 speculation, and greedy decoding |
| Instrumentation | Temporary environment-gated logs recorded proposals, verification choices, committed token IDs, post-penalty top logits, and packed forward shapes; all instrumentation was removed after the investigation |

An explanatory request with 128 input tokens produced the same complete
499-token response under both settings, demonstrating that a kernel change
does not necessarily change the selected tokens.

## Investigation

### 1. Reproduce the Acceptance Shift with Less Work

The investigation began with an observation from the
[continuous-batching benchmark](continuous_batching.md#unexpected-draft-acceptance-shift):
the same fixed-4 speculative model pair reported different acceptance totals
when the active request limit changed from 1 to 2.

We retained its warm-up, prompts, arrival order, paged cache, and fixed-4
policy, but ran only the 2-request workload. At 128 output tokens, serial and
continuously batched execution produced identical response hashes and
acceptance totals:

| Role | Accepted/proposed in both cases |
|---|---:|
| Anchor | 83/233 |
| Follower | 85/211 |

At 256 tokens, both responses diverged and their statistics changed:

| Active request limit | Anchor accepted/proposed | Follower accepted/proposed |
|---:|---:|---:|
| 1 | 154/485 | 166/432 |
| 2 | 161/484 | 168/434 |

Summing the temporary per-verification trace reproduced every final total.
This ruled out draft-statistics accounting as the source of the difference.

### 2. Find the First Different Decision

For each verification, temporary logs recorded the generated-token offset,
the 4 draft token IDs, the target choices, the accepted count, the replacement
token, and the committed tokens.

- The anchor output was identical through zero-based token position 158. Both
  cases then examined the same draft proposal and accepted its first token, but
  the target chose token 476 under serial admission and token 323 under
  continuous batching. The first different output token was at position 159.
- The follower output was identical through position 241. Both cases examined
  the same proposal and accepted its first 3 tokens, but the target then chose
  token 59005 under serial admission and token 13314 under continuous batching.
  The first different output token was at position 242.

The draft proposals were identical at both first divergences. The target model
made the first different decision while verifying those proposals, so the
cause was target-verification sensitivity rather than a draft-model proposal
or statistics defect.

### 3. Inspect Only the Decisive Logits

We then logged post-repetition-penalty logits only at the anchor's first
different position. The two candidates were nearly tied:

| Execution history | Selected token | Selected logit | Runner-up token | Runner-up logit | Margin |
|---|---:|---:|---:|---:|---:|
| Serial admission | 476 | 25.571232 | 323 | 25.565775 | 0.005457 |
| Continuous batching | 323 | 25.562220 | 476 | 25.549618 | 0.012602 |

A change of only a few thousandths in the relative logits reversed the greedy
argmax. Every later token was then conditioned on a different prefix, allowing
the outputs and speculative-acceptance totals to diverge further.

The follower had already finished when this anchor decision was made. The
immediate target forward contained only the anchor's 4-token verification.
The numerical difference must therefore have entered the anchor's state
earlier and persisted through its KV cache; the flip did not require another
request to be present in that exact iteration.

### 4. Connect the Difference to Packed Shapes

Earlier in the continuously batched run, the first mixed target forward packed:

- a 120-token follower prefill; and
- a 4-token anchor verification.

Transformer linear layers therefore received 124 packed rows, while the output
projection selected 1 generation row and 4 verification rows, for 5 rows. With
`MINI_VLLM_METAL_GEMV_MAX_ROWS=4`, both shapes exceed the threshold and use
quantized GEMM. A serial fixed-4 verification has exactly 4 rows and is split
into row-wise GEMV operations.

This gave a concrete mechanism: continuous batching changed the packed tensor
shape, the shape changed the quantized linear kernel, and small numerical
differences entered the cached model state.

### 5. Hold the Kernel Path Constant

Finally, we repeated the shortened comparison with
`MINI_VLLM_METAL_GEMV_MAX_ROWS=1024`. This deliberately routed all relevant
prompt and verification rows through row-wise GEMV regardless of whether the
requests were packed together. It is a numerical control, not a recommended
performance setting.

| Role | Limit 1 accepted/proposed | Limit 2 accepted/proposed | Output hashes matched |
|---|---:|---:|---|
| Anchor | 105/302 | 105/302 | Yes |
| Follower | 109/271 | 109/271 | Yes |

The high-threshold outputs also matched the normal serial outputs. By contrast,
with the default threshold of 4, the batched anchor had already changed from
105/302 to 106/300 over the same 164-token window and its output hash differed.

Holding the kernel path constant restored both token equality and speculative
statistics. No per-request-forward, activation-signature, cache-signature, or
CPU fallback experiments were needed.

## Conclusions

1. The acceptance-rate change was real and correctly counted, but it was a
   consequence of different generated tokens rather than an independent
   continuous-batching effect.
2. Different packed shapes crossed the Metal GEMV threshold and selected
   different quantized matrix-multiplication implementations. Their small
   numerical differences were sufficient to flip a near-tied greedy argmax.
3. Greedy decoding removes sampling randomness; it does not make inference
   numerically invariant across batching shapes or kernels. An early argmax
   flip can cascade into visibly different wording, formatting, and speculative
   acceptance while preserving the broad meaning of a response.
4. The evidence does not show that Metal GEMV produced a wrong result. GEMV and
   GEMM may both be valid floating-point approximations. Establishing relative
   accuracy would require comparing the same captured operation with a
   higher-precision, dequantized reference.

## Appendix: Diagnostic Progression

| Step | Result | What it ruled out or established |
|---|---|---|
| 128-token pair run | Outputs and 168/444 aggregate acceptance matched | Divergence was not immediate |
| 256-token pair run | Both roles diverged; trace totals matched final totals | Statistics accounting defect ruled out |
| Verification trace | Identical proposals, different target replacements | Target verification diverged before the draft proposal |
| Targeted logits | Tokens 476 and 323 reversed across margins below 0.013 | Greedy argmax was numerically sensitive |
| Shape trace | Mixed forward used 124 transformer rows and 5 output rows | Execution crossed the threshold of 4 |
| Threshold 1024 control | Outputs and acceptance totals matched across limits | GEMV/GEMM path difference identified as the cause |
