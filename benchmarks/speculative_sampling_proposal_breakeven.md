# Speculative Sampling Proposal Break-Even

The broader [speculative sampling tradeoff benchmark](speculative_sampling_tradeoff.md)
found that fixed-4 proposals remained slower than target-only decoding even at
100% draft acceptance. This focused follow-up asks whether longer proposals can
amortize the remaining draft and verification overhead, and how much
total-variation (TV) budget is required before they do.

TV=0 preserves the target distribution. TV=1 follows the draft distribution
and accepts every proposed token.

## Reproducibility

| Item | Details |
|---|---|
| Hardware | Apple M2 with an 8-core CPU, 10-core GPU, and 16 GB of unified memory |
| Operating system | macOS 27.0.1 (build 26A434) |
| Models | Qwen2.5 7B Instruct Q4_K_M as the target and Qwen2.5 0.5B Instruct Q4_K_M as the draft; both use the Qwen2.5 7B tokenizer |
| Revisions | [`mini-vllm-rs` `c9fdcad`](https://github.com/lun0522/mini-vllm-rs/commit/c9fdcad537a96da6789fcb386c50df5025124fb5) and [`mini-vllm-eval` `131bcea`](https://github.com/lun0522/mini-vllm-eval/commit/131bcea5d71e22df8e4bc7e66a2b70f661e96493) |
| Procedure | Follow the `speculative-sampling-proposal-breakeven-benchmark` skill in `mini-vllm-eval` and run `speculative_sampling_proposal_breakeven` |
| Configuration | Metal GPU inference with F32 activations, a contiguous 1 GiB target KV cache, a maximum batched token count of 1,024, and `MINI_VLLM_METAL_GEMV_MAX_ROWS=4` |
| Sampling | Temperature 0.7, repetition penalty 1.1 over the last 64 tokens, and seeds 17, 101, and 20261022 |
| Workload | The same 845-token creative-writing prompt as the broader tradeoff benchmark, followed by exactly 256 generated tokens with EOS ignored and streamed output |
| Measurements | Three independent untraced invocations per measured point. Each invocation used a 1-token warm-up and one measured request for each of the three seeds. Results below are medians of the three invocation-level medians |

The endpoint pass measured target-only, draft-only, and fixed proposal lengths
4, 6, and 8 at TV budget 1. Only fixed-6 and fixed-8 were close enough to
target-only to justify selective measurements at budgets 0, 0.5, and 0.75.
Fixed-5 and fixed-7 were not tested. The search stopped after fixed-8 had
adjacent tested budgets on opposite sides of target-only.

## Metrics

- **End-to-end (E2E) latency:** Time from request submission through the final
  streamed response.
- **Decode throughput:** `255 / (E2E latency - time to first token)`.
- **Output per round:** The average number of generated tokens produced by each
  target verification, including the replacement token after a rejection.

## Results

### Controls

| Configuration | E2E (s) | E2E vs. target | Decode (tok/s) | Decode vs. target | Acceptance | Accepted/round | Output/round | Verifications/invocation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Target-only | 20.971 | baseline | 18.21 | baseline | - | - | - | - |
| Draft-only | 3.372 | 83.9% lower | 110.50 | 506.8% higher | - | - | - | - |
| Fixed-4, TV=1 | 24.425 | 16.5% higher | 15.73 | 13.6% lower | 100.0% | 4.00 | 4.00 | 192 |

### Fixed-6

| TV budget | E2E (s) | E2E vs. target | Decode (tok/s) | Decode vs. target | Acceptance | Accepted/round | Output/round | Verifications/invocation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38.840 | 85.2% higher | 8.35 | 54.1% lower | 23.8% | 1.41 | 2.36 | 326 |
| 0.5 | 23.949 | 14.2% higher | 16.27 | 10.7% lower | 71.5% | 4.26 | 4.74 | 162 |
| 0.75 | 21.115 | 0.7% higher (approximately equal) | 19.87 | 9.1% higher | 94.7% | 5.61 | 5.73 | 134 |
| 1 | 20.587 | 1.8% lower (approximately equal) | 20.68 | 13.6% higher | 100.0% | 5.95 | 5.95 | 129 |

### Fixed-8

| TV budget | E2E (s) | E2E vs. target | Decode (tok/s) | Decode vs. target | Acceptance | Accepted/round | Output/round | Verifications/invocation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.406 | 92.7% higher | 7.94 | 56.4% lower | 19.0% | 1.50 | 2.48 | 310 |
| 0.5 | 22.900 | 9.2% higher | 17.50 | 3.9% lower | 63.5% | 5.01 | 5.57 | 138 |
| 0.75 | 18.820 | 10.3% lower | 24.25 | 33.2% higher | 93.4% | 7.39 | 7.53 | 102 |
| 1 | 18.508 | 11.7% lower | 24.94 | 37.0% higher | 100.0% | 8.00 | 8.00 | 96 |

Fixed-8 was the only proposal length with a clear end-to-end win. It was 9.2%
slower than target-only at TV=0.5 and 10.3% faster at TV=0.75, making **0.75 the
lowest tested clearly winning budget**. No budgets between 0.5 and 0.75 were
tested. At TV=1, fixed-8 reduced E2E latency by **11.7%** and raised decode
throughput by **37.0%**.

Fixed-6 did not clearly beat target-only: its nominal 1.8% latency reduction at
TV=1 was within run-to-run variability. Fixed-4 was 16.5% slower at TV=1.

At TV=0.75, fixed-8 produced 7.53 output tokens per verification round and
required 102 rounds across the three requests, compared with 5.73 tokens per
round and 134 rounds for fixed-6. Both used the regular quantized-matmul path
for full proposals, so the result is consistent with fixed-8 benefiting from
fewer verification rounds. Comparisons with fixed-4 also cross the
[kernel-dispatch boundary described below](#verification-kernel-paths).

The invocation- and seed-level measurements supporting these classifications
are in the [latency evidence appendix](#appendix-latency-evidence). Fixed-8 at
TV=0.75 and TV=1 was faster for every seed, while fixed-6 changed sign by seed.

### Why draft-only speed does not guarantee a win

Draft-only decoding measures the small model by itself. Speculative decoding
still runs the target model and adds autoregressive draft generation,
acceptance and correction sampling, and KV-cache bookkeeping. Even at TV=1,
where every draft token is accepted, the reduction in target calls must repay
this additional work.

For this 256-token workload, target-only prefill produces the first token and
then performs 255 sequential decode steps. A fixed-`k` speculative round
replaces several of those steps by generating up to `k` draft tokens and
evaluating them together in one target verification pass.

The measured round counts show the useful amortization directly:

| Configuration | Acceptance | Output/round | Verifications/invocation | E2E (s) |
|---|---:|---:|---:|---:|
| Fixed-4, TV=1 | 100.0% | 4.00 | 192 | 24.425 |
| Fixed-6, TV=1 | 100.0% | 5.95 | 129 | 20.587 |
| Fixed-8, TV=1 | 100.0% | 8.00 | 96 | 18.508 |
| Fixed-6, TV=0.75 | 94.7% | 5.73 | 134 | 21.115 |
| Fixed-8, TV=0.75 | 93.4% | 7.53 | 102 | 18.820 |

The counts cover three requests, so fixed-8 at TV=1 performs 32 verification
rounds per request instead of 255 target-only decode steps. Fixed-4 still
performs 64 rounds per request, while also paying for four draft steps and
speculative bookkeeping in every round. This measured work explains why a
standalone draft model that is about 6.2 times faster end to end does not make
fixed-4 speculation faster: standalone draft E2E is not the draft-time
component inside speculation, and speculation still evaluates the target.

The analysis therefore uses measured round counts rather than a fitted cost
model.

### Verification kernel paths

The benchmark set `MINI_VLLM_METAL_GEMV_MAX_ROWS=4`. At the tested
`mini-vllm-rs` revision, quantized Metal matrix multiplications with two to
four activation rows are split into separate row-wise GEMV calls; inputs with
more than four rows use Candle's regular quantized-matmul path. Target
verification has one activation row per proposed token.

Consequently, fixed-4 verification uses `metal-rowwise-gemv`. Full fixed-6
and fixed-8 proposals use `quantized-matmul`; only a shortened final proposal
of four or fewer tokens can use row-wise GEMV. At TV=1, fixed-6 has 129 rounds
for 768 output tokens because each request ends with a four-token proposal,
whereas fixed-8 divides 256 evenly and has 96 full eight-token rounds.

The fixed-4-to-fixed-6/8 comparison therefore crosses a kernel-dispatch
boundary as well as increasing the number of output tokens per round. Code
inspection established these dispatch paths.

## Output Samples

The [complete output appendix](#appendix-complete-prompt-and-output-samples)
contains the full prompt and every 256-token output for target-only, draft-only,
and fixed-8 at TV=0, 0.5, 0.75, and 1 across seeds 17, 101, and 20261022. These
texts were collected separately from the performance runs.

Across all three seeds:

- Target-only and TV=0 speculation most consistently follow the requested
  condensation, abandoned-greenhouse, and investigation setup. Individual
  details differ because TV=0 preserves the target distribution, not an
  identical sampled sequence.
- Fixed-8 at TV=0.5 has consistency problems in all three samples, but the
  particular failures are sample-specific: seed 17 changes character gender
  and roles, seed 101 contains awkward local phrasing, and seed 20261022 adds a
  daughter and an unexplained world beyond the window.
- Fixed-8 at TV=0.75 also shows recurring prompt-adherence or continuity
  problems across all three seeds. Unseen footsteps and “Earth’s infrared”
  occur only in seed 101; other seeds instead contradict the power history,
  change pronouns, or introduce extra crew and Earth-like scenery.
- Fixed-8 at TV=1 has problems in at least two seeds: seed 17 introduces and
  renames Ilyan's sister, while seed 101 swaps Mara's identity and invents the
  terminal-command relationship. Seed 20261022 is more locally coherent but
  does not reach several required plot beats within the 256-token window.
- Draft-only drifts from the supplied physical and narrative constraints in
  every seed. Diesel fuel, a brass latch, and a dustpan are specific to seed
  101; the other seeds introduce such details as a wheel-and-ladder scene,
  lunar wind, or contradictory power and condensation history.

These samples support qualitative comparison only. They do not establish a
safe TV budget or make TV distance a measure of semantic quality.

## Conclusions

1. Fixed-8 clearly beat target-only at TV=0.75 and TV=1. Fixed-6 was
   approximately equal at those budgets, and fixed-4 was slower at TV=1.
   Fixed-8 required fewer verification rounds; comparisons with fixed-4 also
   cross a verification-kernel boundary.
2. Fixed-8 was slower at TV=0.5 and faster at TV=0.75, making TV=0.75 the
   lowest tested clearly winning budget. Budgets between them were not tested.

## Appendix: Latency Evidence

Main-table values are medians of the three invocation medians below. CV shows
their run-to-run variability.

| Configuration | Invocation medians (s) | CV |
|---|---:|---:|
| Target-only | 20.971, 20.829, 21.085 | 0.61% |
| Draft-only | 3.291, 3.485, 3.372 | 2.88% |
| Fixed-4, TV=1 | 24.348, 24.551, 24.425 | 0.42% |
| Fixed-6, TV=0 | 37.983, 38.840, 38.915 | 1.34% |
| Fixed-6, TV=0.5 | 24.043, 23.917, 23.949 | 0.27% |
| Fixed-6, TV=0.75 | 20.909, 21.115, 21.345 | 1.03% |
| Fixed-6, TV=1 | 20.587, 20.584, 20.869 | 0.79% |
| Fixed-8, TV=0 | 40.406, 40.164, 40.467 | 0.40% |
| Fixed-8, TV=0.5 | 22.466, 23.022, 22.900 | 1.28% |
| Fixed-8, TV=0.75 | 18.701, 18.820, 18.837 | 0.39% |
| Fixed-8, TV=1 | 18.655, 18.508, 18.145 | 1.42% |

A result is clearly faster when its invocation range is below the target-only
range, clearly slower when it is above, and approximately equal when the ranges
overlap. The target-only range was 20.829–21.085 seconds.

Per-seed values were reconstructed from the preserved request order and are
medians across the three invocations. Parentheses show the difference from
target-only for the same seed.

| Configuration | Seed 17 | Seed 101 | Seed 20261022 |
|---|---:|---:|---:|
| Target-only | 20.865 s | 21.220 s | 20.971 s |
| Draft-only | 3.298 s (-84.2%) | 3.511 s (-83.5%) | 3.372 s (-83.9%) |
| Fixed-4, TV=1 | 24.348 s (+16.7%) | 24.468 s (+15.3%) | 24.520 s (+16.9%) |
| Fixed-6, TV=0 | 42.389 s (+103.2%) | 37.185 s (+75.2%) | 38.840 s (+85.2%) |
| Fixed-6, TV=0.5 | 23.949 s (+14.8%) | 24.160 s (+13.9%) | 23.216 s (+10.7%) |
| Fixed-6, TV=0.75 | 21.115 s (+1.2%) | 21.440 s (+1.0%) | 20.400 s (-2.7%) |
| Fixed-6, TV=1 | 20.882 s (+0.1%) | 20.701 s (-2.4%) | 20.265 s (-3.4%) |
| Fixed-8, TV=0 | 40.406 s (+93.7%) | 39.598 s (+86.6%) | 41.691 s (+98.8%) |
| Fixed-8, TV=0.5 | 23.193 s (+11.2%) | 22.900 s (+7.9%) | 22.096 s (+5.4%) |
| Fixed-8, TV=0.75 | 18.820 s (-9.8%) | 19.542 s (-7.9%) | 18.784 s (-10.4%) |
| Fixed-8, TV=1 | 18.658 s (-10.6%) | 18.183 s (-14.3%) | 18.293 s (-12.8%) |

## Appendix: Complete Prompt and Output Samples

This appendix preserves the complete prompt and all 18 generated outputs used for the follow-up report’s qualitative audit. The output-only collection used `mini-vllm-rs` revision `c9fdcad` and the submitted `mini-vllm-eval` revision `131bcea`; its timing results are not included in the performance analysis. Each output stops at exactly 256 generated tokens, so endings may be incomplete.

### Collection Command

```sh
MINI_VLLM_SAMPLING_PROPOSAL_BREAKEVEN_CASES='target,draft,8:0,8:0.5,8:0.75,8:1' \
  .venv/bin/python3 main.py --benchmark speculative_sampling_proposal_breakeven \
  2>&1 | tee /private/tmp/mini-vllm-sampling-proposal-output-audit.log
```

### Prompt

> Write an original literary science-fiction scene of at least 1,200 words. The scene takes place
> in an abandoned greenhouse on the Moon while a station mechanic and an archivist search for the
> source of a repeating distress signal. Begin with the mechanic discovering fresh condensation on
> a window even though the greenhouse has been without power for twenty years. Let the characters
> disagree about whether to restore the local system, gradually reveal a personal reason the
> archivist recognizes the signal, and make the restoration create an unexpected but nonviolent
> danger.
> 
> Use the following background as constraints, not as text to quote. The greenhouse belonged to a
> failed agricultural research wing called Selene Nine. Its long central aisle is bordered by dry
> hydroponic beds, collapsed irrigation tubes, and cabinets containing seed envelopes whose labels
> have faded. A transparent roof looks toward Earth, but electrostatic dust has left only irregular
> patches of blue reflected light. Emergency doors divide the greenhouse into three pressure zones.
> The first zone is open to the station corridor, the second retains a thin stale atmosphere, and
> the third has not been opened since the wing was abandoned. An old maintenance terminal near the
> second door controls power, ventilation, irrigation, and the door interlocks. It can restore only
> one subsystem at a time until its batteries recharge.
> 
> The mechanic, Mara Venn, is practical, tired, and accustomed to repairing systems with incomplete
> records. She distrusts unexplained signals because a false evacuation alarm once injured members
> of her crew. The archivist, Ilyan Sato, is careful with objects but evasive about his own history.
> Years earlier his older sister recorded navigation beacons for remote lunar shelters. The
> repeating signal contains a cadence he associates with her, although he initially describes it
> only as an obsolete checksum. Neither character knows whether the resemblance is intentional,
> accidental, or the result of damaged equipment. Do not confirm a supernatural explanation.
> 
> Build the scene through physical investigation rather than exposition. Allow Mara and Ilyan to
> notice different details and draw conflicting conclusions. Their dialogue should include pauses,
> unfinished thoughts, and practical discussion of pressure seals, battery charge, and signal
> routing, but it should remain understandable without technical background. Reveal Ilyan's
> connection to the signal gradually through his choices and reactions before he explains it. Give
> Mara a credible reason to activate the terminal despite her distrust. When power returns, make
> the greenhouse respond through several linked changes rather than a single dramatic switch.
> 
> Let the pacing alternate between quiet observation and short bursts of purposeful movement.
> Objects should carry evidence of ordinary people who once worked there, but invent those traces
> freely rather than turning them into a historical explanation. The characters may disagree about
> what a damaged log entry, tool, photograph, or handwritten label means. Give each character at
> least one moment in which an assumption proves incomplete. Their relationship may soften under
> pressure, but avoid romance, speeches about trust, or a sudden confession that settles their
> disagreement. Leave room for ambiguity in both their motives and the greenhouse's behavior.
> 
> The resulting danger must be immediate and unexpected but nonviolent: for example, shifting
> pressure, rapidly spreading condensation, awakened machinery, obscured visibility, or doors
> following an old quarantine routine. It must arise logically from the neglected greenhouse and
> force the characters to cooperate. Do not introduce an attacker, alien creature, weapon, rescue
> team, or sudden external disaster. Do not resolve whether the distress signal was deliberately
> sent. End this excerpt while the characters are still confronting the consequences of their
> decision, not with a summary or a neat explanation.
> 
> Use vivid sensory details and a restrained emotional tone. Recur to dust, reflected Earthlight,
> and dormant seeds, changing what those images mean as the scene develops. Keep the viewpoint
> close to Mara without using first person. Avoid headings, an outline, epigraphs, flashback
> sections, or authorial commentary. Do not mention these instructions. Continue the scene well
> beyond 512 model tokens rather than ending early.

### Target only, seed 17

> The greenhouse was a ghostly shell of what it once had been, its central aisle now a desolate corridor lined with dry hydroponic beds and collapsed irrigation tubes. Mara Venn hesitated at the emergency door, her hand resting on the cold metal frame. The stale air inside pressed against her like an old memory, thick and heavy. She took a deep breath, feeling the dust particles floating in the dim light reflected from the broken roof.
> 
> Mara stepped into the second pressure zone, her eyes scanning the room for any sign of life. The faint blue patches on the ceiling were the only reminders that there was once liquid water here. The air felt unnaturally still, and she could hear the distant hum of machinery in the background—a gentle reminder of a time when this place had been alive.
> 
> "Did you find anything?" Ilyan Sato’s voice broke through her reverie. He stood at the far end of the aisle, his fingers brushing against a faded label on one of the cabinets.
> 
> Mara shook her head. "Nothing. Just dust and silence." She walked over to where he was standing, noting that his hand lingered longer than necessary before it moved away from the cabinet. The label had once been bright green, now faded to a dull

### Target only, seed 101

> The greenhouse was a vast expanse of silence, its air stale and thick with dust that seemed to cling to every surface like an invisible film. Mara Venn stepped through the first pressure zone, her boots crunching on loose dirt as she surveyed the interior. The central aisle stretched into darkness, each hydroponic bed now barren and covered in electrostatic dust that sparkled faintly under the reflected light from Earth. A few tools lay scattered about, rusted and forgotten.
> 
> Mara approached a window, her eyes narrowing as she noticed something peculiar. Condensation had formed on the glass, even though the greenhouse had been without power for twenty years. She reached out to wipe it away with a gloved hand, but before her fingers could touch, a spot of moisture shimmered and then vanished.
> 
> Ilyan Sato followed behind her, his movements deliberate and cautious. "Mara," he said softly, "do you think this means...?"
> 
> "Means what?" she replied, meeting his gaze. "That someone's been here recently? Or that something’s changing? I don’t know, but if it is, we need to be ready."
> 
> Ilyan nodded, his expression thoughtful. He moved closer to the maintenance terminal near the second door. The room

### Target only, seed 20261022

> Mara Venn stepped into the abandoned greenhouse, her boots crunching on layers of electrostatic dust that clung to her spacesuit like tiny prisms of moonlight. The air was stale and cold, barely enough for her lungs to draw a breath. Her eyes adjusted to the dim blue light reflected from the transparent roof, which looked like a faded photograph of Earth. She had been here before, many times, but never so late at night.
> 
> The greenhouse’s central aisle, once bustling with activity, was now a quiet expanse of dry hydroponic beds and collapsed irrigation tubes. As she walked down the aisle, her hand brushed against something solid—condensation on the window.
> 
> “Unbelievable,” Mara muttered to herself, wiping away the mist. “It hasn’t been powered in two decades.” She had seen this greenhouse many times before, but never with condensation. It should have been a clear, cold vacuum. Her fingers traced the fine lines of water droplets, each one marking a moment when something had escaped.
> 
> Ilyan Sato followed her into the greenhouse, his steps softer than hers. He wore gloves that muffled the sound of his movements, but still, Mara could almost hear them. His eyes flicked across the aisle as

### Draft only, seed 17

> ### **The Unfinished Journey**
> 
> The old maintenance terminal is a dimly lit beige box in the heart of the abandoned greenhouse on the Moon's lunar surface. Its first appearance was an obscure blue light, marking its location and the end of the station corridor. A few moments before, it had been a bright indicator that something significant was about to unfold.
> 
> The mechanic, Mara Venn, whose eyes were weary from her thirty-plus years in the field, found herself standing under the night sky, her right hand gripping a worn front wheel frame on a step ladder. The distant lights of the starless cosmos seemed to come alive beneath her feet, their last flicker extinguished by the screens of an old computer.
> 
> As she turned, the mechanics panel flickered with each movement of the terminal, its light tending towards a soft yellow that suggested it was still recovering from some recent malfunction. She hesitated for a moment before pulling herself up on the ladder and stepping into the terminal's dimly lit interior.
> 
> Inside, the familiar machinery hummed in an unbroken rhythm—its gears settling down with less noise than she'd expected but no change in energy output to justify it.
> 
> "Ready," she muttered, her voice hoarse from use. She knew what she needed: a

### Draft only, seed 101

> ### **The Greenhouse's Call**
> 
> The greenhouse on the Moon was a forgotten relic from Selene Nine. Its central aisle had been empty since its abandonment nearly twenty years ago, except for a single window that once glimmered with the remnants of an evacuated crew member’s faint electronic signal.
> 
> Mara Venn and Ilyan Sato wandered the dusty corridors, their footsteps echoing against the rough surface like the tiles in a failed studio suite. The greenhouse had been transformed into a makeshift laboratory, its central aisle now lined with old tools, fragile papers, and an incongruous smell of rotting fruit and aged diesel fuel.
> 
> Mara took a deep breath, her curiosity piqued by the intermittent noise coming from the unlit second door, a symbol of the forgotten crew. She pushed open the heavy brass latch, and a faint whiff of stale air wafted in. Her eyes adjusted to the dim light, checking for signs of damage.
> 
> She saw a small dustpan sitting on the floor, its edges worn but still reflective. "Welcome home," she said softly, her voice barely audible over the hum of the greenhouse's machinery. She turned and walked towards the door, taking a tentative step inside. Ilyan caught up to her behind the closed

### Draft only, seed 20261022

> The greenhouse on the Moon's edge was a somber testament to its past glory, now silent under the dust and rot of time. Its central corridor had been planted with seed envelopes, their labels faded but not obliterated by the relentless wind. The maintenance terminal nestled in the second zone, its glass door slightly ajar like a secret entrance, held the promise of salvaging either the archive or the signal.
> 
> The greenhouse was an anomaly, a forgotten sanctuary on Earth's lunar frontier, a testament to the sparsest memory of life in our solar system. It had been abandoned decades ago, its secrets buried beneath the vast expanse of the Moon, left to rot and perish like the rest of it. But Mara Venn, the mechanic, found herself drawn to the window that faced out into the corridor, where a patch of blue reflected light remained from the long nights when the greenhouse had been without power.
> 
> "Let's go," she said softly, her voice barely above a whisper. "I find something odd here."
> 
> Mara pushed aside the dust covering the window and began to examine it more closely. The condensation that had always greeted her was now almost undetectable, but the faint glow of stars within the pane suggested that perhaps the greenhouse might still be powered somehow

### Fixed 8 / TV=0, seed 17

> The air was still, and the only sound came from Mara Venn’s breathing as she stood by one of the emergency doors. The door itself was made of reinforced metal, now covered in a layer of dust that seemed almost to dance under the faint light filtering through the greenhouse's broken roof. The stale air inside felt like it hadn't moved for years, but Mara knew better than to trust her senses alone.
> 
> She had been here many times before, checking on the Selene Nine agricultural wing for any signs of life or malfunction. But today was different. She noticed something odd as she reached out a hand towards one of the hydroponic beds—there was condensation forming on the window, even though the greenhouse had been without power for twenty years.
> 
> Mara’s fingers ran across the glass, feeling the dampness like a ghostly touch. “I can’t believe this,” she muttered to herself, her voice barely audible over the quiet hiss of her own breath.
> 
> “Did you just say something?” Ilyan Sato’s voice came from behind her, soft and hesitant. He had been following Mara for what felt like hours, his movements slow and deliberate. His eyes traced Mara’s every action, as if he were afraid to miss a detail.
> 
> M

### Fixed 8 / TV=0, seed 101

> The air in the greenhouse was thick with electrostatic dust, its particles clinging to the faded labels of the hydroponic beds and casting irregular blue patches across the roof panels. Mara Venn stood at the first emergency door, her hand resting on the cold metal frame. She had been here before, but never this late in the evening. A faint condensation had gathered along the window pane, its droplets shimmering under the reflected light from Earth.
> 
> Mara sighed and rubbed a finger across the glass. "It's just water vapor," she muttered to herself. "The greenhouse hasn't had power for twenty years."
> 
> Ilyan Sato appeared beside her, his eyes scanning the scene with a practiced caution. He reached out and touched one of the labels, his fingers brushing against the faded text. "Look at this," he said softly.
> 
> Mara turned to him. "What is it?"
> 
> "It's not just condensation. Look." Ilyan pointed to another window where small droplets were forming in an irregular pattern. "It's a pulse, like someone is turning on and off a humidifier."
> 
> Mara frowned. "We don't have power here," she said firmly. "Stop wasting time with speculation and let’s focus on what we can

### Fixed 8 / TV=0, seed 20261022

> The greenhouse’s floor creaked under Mara Venn's boots as she moved slowly down the central aisle. The air was thick with dust and the faint scent of decaying plants. She had been here before, or at least in its pressurized sections, but never to this extent. The hydroponic beds were now bare, their once-lush vegetation replaced by a mat of dust and decay. The only sound was her boots on the hard-packed soil.
> 
> Ilyan Sato followed behind, his steps lighter. He moved with a careful precision that made Mara suspicious. "Did you find anything?" she asked, eyeing him warily.
> 
> "Nothing yet," he replied. His eyes flicked towards the emergency door, the one closest to her. "But I’m not sure we should be looking."
> 
> Mara stopped and turned to face him. "What do you mean?"
> 
> "Ilyan, I’ve been through these files enough times. There’s no reason for a distress signal here. It could be a false alarm. Remember what happened last year?"
> 
> Mara sighed. She recalled the incident vividly—a simulation gone wrong had triggered an evacuation that resulted in some of her crew being injured by the rapid depressurization. "I know, but something

### Fixed 8 / TV=0.5, seed 17

> The greenhouse, once a vibrant incubator for Selene Nine's advanced research, now lies in ruins under Earth’s dim reflection. The central aisle is derelict, its hydroponic beds shifting with a gentle breeze that commutes the air through dust-laden vents. Mara Venn, the mechanic and station director, knurls an electrical cable through the broken-open doorway, his fingers brushing against the rough fabric of the overalls he wears. "These are not new," he murmurs, his voice riven with fatigue.
> 
> "If we can fix this, it will be a milestone in our work," Ilyan Sato, the archivist, says, his eyes narrowing as he moves closer to the now-empty terminal.
> 
> "It's been a long time since you last checked on the greenhouse," Mara replies, shifting from foot to foot. His voice carries an unspoken accusation that seems to echo through the empty building.
> 
> "I've lost track of time," Ilyan sighs, his eyes fixed on a faint patch of blue light reflecting off the ancient glass panels that once filtered light into the greenhouse's interior. "I know it's been twenty years since you last checked."
> 
> Mara pauses for a moment, tracing the pattern of the dust-dusted vents with his

### Fixed 8 / TV=0.5, seed 101

> The greenhouse's central aisle was a labyrinth of dry hydroponic beds and collapsed irrigation tubes. The station corridor led to the second zone, where a thin stale atmosphere clung to each door frame like a ghostly embrace. Mara Venn, the mechanic, had caught her hand on the emergency door, fingers tracing the shape of the textured metal. She sighed softly as she stepped inside, the faint sound of soil and decay from within.
> 
> "Pulling double duty," she muttered, her voice barely audible over the further silence of the greenhouse.
> 
> Ilyan Sato followed her eye, his gaze drifting through the faded seed envelopes that still lay open on the windowsill. He lingered there for a moment, studying each one, before he spoke. "It's not an evacuation alarm," he said softly, his voice caught in its usual evasiveness. "It's something else."
> 
> Mara nodded, her face a mask of contemplation. She had seen it once on the other end of the line: a false signal—a glitch that sent a distress call without the requested response.
> 
> "Is there any reason I should restore this?" she asked, glancing at her watch. "Power still needs to be restored."
> 
> Ilyan considered his words carefully. The terminal

### Fixed 8 / TV=0.5, seed 20261022

> The greenhouse in the abandoned Selene Nine laboratory is a living proof of technology left to rot. The central deck is like a forgotten aisle, flanked by dry hydroponic beds and collapsed irrigation tubes that have not been replaced for decades. Beneath this sterile dome lies a dark interior where no light penetrates except through a transparent roof that reflects only irregular blue-streaked patches of light.
> 
> The greenhouse has one window - the kind Mara Venn uses to see into the future, and she notices it is different today - instead of an unlit window in the past. The greenhouse had been without power for twenty years. But now, another world appears on Earth's horizon: a station room that only two other people could have lived inside.
> 
> The window is old, cracked with age and dust, but it still reflects light. It was there when Mara first arrived at the laboratory; she remembers it as if it were the last time she saw her own eyes once more. She had brought her daughter from Earth to see this greenhouse, and now they would be among the first to experience a signal sent by this system.
> 
> "Is that condensation?" Ilyan Sato's voice suggests disbelief in a sentence he didn't quite finish.
> 
> "The old maintenance terminal," Mara replies with

### Fixed 8 / TV=0.75, seed 17

> The greenhouse, once a vibrant incubator for innovations and dreams, now stands as a desolate relic of its former glory, its windows shimmering with dust that has captured the light from the distant moon. The central aisle is narrow enough to accommodate two people, each standing at the edge in their own contemplative spaces. Mara Venn, the mechanic, her eyes heavy with wear and tear, listens as Ilyan Sato approaches, his voice steady despite the weight of this abandoned structure.
> 
> "I've noticed a slight change," Mara starts with cautious observation, compared to the mechanical precision she's accustomed to considering. "There's no sign of life in the windows; they’ve been empty since last power was restored."
> 
> Ilyan Sato, one of the few survivors left from Selene Nine, nods, his eyes never leaving the dusty ground. "This is strange," he says, his voice not a breath away from quietude. "What's happening here?"
> 
> Mara fiddles with her tools, revealing layers of information that leave him unsure. "The signal," she begins formally, as if to set expectations but ultimately avoiding a definitive explanation. "It seems to be altered or dormant."
> 
> "What does the signal mean?" Ilyan asks, his mind already moving

### Fixed 8 / TV=0.75, seed 101

> The greenhouse's central aisle was a labyrinth of dry hydroponic beds and collapsed irrigation tubes. The station corridor led to an emergency door that opened into the third zone, where Ilyan Sato and Mara Venn stood silent, their voices barely audible over the cacophony of distant machinery and unseen footsteps.
> 
> Mara glanced at the ceiling, her eyes widening as she saw the first stained layer of condensation on a window protecting the greenhouse. The roof shifted slightly, revealing a thin sliver of blue light streaming through—Earth’s infrared, indicating that it was not the end of this system but only an interruption in its normal operation.
> 
> "Maybe we should take out the old batteries," Mara suggested, her voice barely above a whisper.
> 
> Ilyan nodded slowly. "You're right. There are no more logs left to restore. Maybe I'll just try to power off before it gets too dark."
> 
> Mara nodded again, feeling a moment of unease but forced into silence by the need to conserve energy for the systems that were still in operation. She turned to look at her companion.
> 
> "You know what? I think we should work together to restore this," she said, her voice barely above a whisper.
> 
> Ilyan's expression remained terse. "That’s

### Fixed 8 / TV=0.75, seed 20261022

> The greenhouse in the abandoned Selene Nine laboratory is a living proof of technology's triumph over nature, its empty windows reflecting an airless landscape. Mara Venn, her eyes glazed with exhaustion, turns to find his crewmates huddled around a small table under the last remaining electric light, their curiosity piqued by the cloudless atmosphere and the eerie silence that lingers in the post-apocalyptic twilight.
> 
> Ilyan Sato, his voice thick with tension, rises from his wooden bench. The station corridor is now open to the world, but it seems as though a new day has started on this barren green earth. A single window beyond his own chills him, its glass pane showing only a pale blue line against the black of the night sky.
> 
> "Have you noticed anything unusual?" Mara asks, her voice heavy with both concern and frustration.
> 
> Ilyan hesitates for a moment before beginning to clarify. "There's something... strange about this greenhouse," he begins, his voice tinged with curiosity as well as fear.
> 
> Mara ignores his words, her mind already making the connection. She leans against the wall, her hand reaching out to grab at any sign of the signal. The mechanics have been working for years, cleaning and repairing systems that once seemed

### Fixed 8 / TV=1, seed 17

> The greenhouse, once a vibrant incubator for innovations and dreams, now stands as a desolate relic of its former glory, its windows shimmering with dust that has captured the light from the distant moon. The central aisle is narrow enough to accommodate two people, each standing at the edge in their own isolation. Mara Venn, the mechanic, the tired heart of the station crew, takes one step forward on this narrow corridor. Her boots scrape against the cold stone floor as she steps into a zone where her journey has ended. The second door opens with an eerie quiet, opening to a sterile atmosphere that could only belong to a failed agricultural research wing.
> 
> Ilyan Sato, the archivist and historian of the greenhouse, leans against the thin metal frame, his eyes fixed on the still lighted window above him. His older sister's beacon device, now a relic of the past, rests in the same spot as it had been from the day he first set foot on the Moon. The information that could have led them to this moment is as hidden as time itself.
> 
> "Ilsan was her name," Ilyan says in a whisper, his voice barely audible over the distant strains of an unspoken conversation. "We never knew who she was until we discovered her

### Fixed 8 / TV=1, seed 101

> The greenhouse's central aisle was a labyrinth of dry hydroponic beds and collapsed irrigation tubes. The station corridor led to an emergency door that opened into the third zone, where Ilyan Sato and Mara Venn worked as maintenance engineers under the command of the old maintenance terminal near the second door.
> 
> Mara strode through the dusty corridors, her footsteps echoing off the concrete floor. She saw a triangular structure rising from the ground, a reflective surface over the windows that reflected light in strange ways. "It's supposed to be an emergency exit," she muttered under her breath, though she already knew the signal was coming.
> 
> Ilyan stepped aside, his eyes scanning the scene with a mixture of trepidation and curiosity. The greenhouse had been abandoned without being repaired for over twenty years, but he recognized its presence as unlikely. "It's not like this," he said, his voice low and calm.
> 
> "This is Selene Nine, one of the failed agricultural research wings that were once part of an artificial lunar colony," Mara explained. "The greenhouse was a critical system that controlled water and nutrients for our plantations."
> 
> "I'm Ilyan Sato, the archivist from the station," she added, her voice steady but still tinged with interest.
> 
> "

### Fixed 8 / TV=1, seed 20261022

> The greenhouse in the abandoned Selene Nine laboratory is a living proof of technology's triumph over nature, its empty windows reflecting an airless landscape. The ground beneath leather-bound seed envelopes holds seeds that could be transplanted to other planets, but they are not yet ripe enough to grow into actual plants.
> 
> Inside, the central aisle is a cruel mockery of human craftsmanship: dry hydroponic beds, collapsed irrigation tubes, and cabinets filled with labels that have faded to memory. The maintenance terminal near the second door seems like an unmarked relic from the past, its windows paneled by dust. A slight flicker of blue light reveals itself in the transparent roof above, a stark reminder of the station's high-tech past.
> 
> Mara Venn, a mechanical engineer with a graying face and eyes filled with technical expertise, studies the window as if it were a tear in her favorite film screen. She notices that the glass surface is even now, untouched by time, but her gaze never leaves the pane of darkness below. What she sees isn't distress; it's a signal.
> 
> "Could this be a glitch?" Mara murmurs to herself, her mind racing with potential explanations for why some systems are not functional as they should be. The signal is faint and imperceptible

