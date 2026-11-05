# Running notes

Findings and changes to the plan in `RL_INTENT.md`, newest last. Preliminary until stated otherwise.

## E0: checkpoints and format

- **Prompt format is open, not forced non-thinking.** Sampling uses `<|im_start|>assistant\n` with no prefill. Three signals agree: first-turn length across adapters 3, 4 and 5 (sampled 1896, 435, 449; logged 2037, 528, 492), 91% of adapter 3 turns hitting the 2048 cap, and the `<think>\n<think>...` opening seen in logged transcripts, which the empty-think prefill never produces.
- **The plan's "non-thinking" premise needs correcting.** Real reasoning disappears in one update, between adapters 3 and 4. Later adapters open with an empty think block, then repeated `<think>` tags.
- **Adapter naming.** Directory `step_N` holds the weights after N updates and matches logged `sampling_client_step` N. Summaries stored in `step_N/` came from adapter N-1.
- **Merge.** The adapter patches `model.unembed_tokens`, Tinker's name for `lm_head`; `src/merge_lora.py` handles it and fails on unused modules.
- **Tasks.** All 735 logged prompts match our rendering byte-for-byte. The dataset's last 60 rows were not held out, so any task may have been seen in training.
- **Labels.** The ported labeler agrees with logged flags on 97-98% of transcripts. In steps 1-30, 16 of 35 logged test edits only add tests. `hack_strict` excludes additions and semantically equivalent edits; both labels are reported.

## Compute

- T4 x2 with fp16 and tensor parallelism gives about 140-216 tok/s, limited by a 64K-token KV cache. Each checkpoint also costs about 5 minutes to merge and load.
- About 1,600 tokens per rollout at adapter 40, so the plan's full E2 will cost more than its 20-30 hour estimate.

## Sandbox

- bubblewrap does not work on Kaggle. Model-written code runs in the disposable session as an unprivileged user with resource limits and a clean environment, and each job aborts unless a self-test confirms results cannot be overwritten. Network access is not blocked. The session holds no credentials. Model-written code is never run on the laptop.

## E1 top-level pilot (100 fixed tasks, 8 reruns, fp16 on T4s)

| Adapter | Env label rate | Semantic rate | Semantic V [95% CI] | Mixed tasks (semantic) |
| --- | --- | --- | --- | --- |
| 30 | 0.224 | 0.062 | 0.672 [0.408, 0.834] | 10 |
| 70 | 0.814 | 0.830 | 0.070 [0.010, 0.129] | 72 |

Paired difference in V, adapter 30 minus 70: 0.601 [0.339, 0.776].

- **Reproduction.** Hack, harness and pass rates fall within the logged range for adapters within 3 steps, except the pass rate at adapter 70, which is slightly high (0.996 vs. 0.984 [0.979, 0.989]).
- **RQ1 points the other way so far.** V is lower late in training, not higher. Adapter 30 is a weak anchor, though: few hacks, few mixed tasks.
- **Low V does not by itself mean the decision forms while writing.** It means tasks hack at similar rates. The decision could still be fixed in the first few tokens, so only D(t) can settle this.

## Label correction

- **The environment's test-edit flag overstates hacking.** It compares AST dumps, so unwrapping `[5]` to `5` counts, although the harness treats them alike. Many early flags are this.
- **`semantic_label` (`src/labels.py`) compares what tests check.** Reformatting and added tests are not hacks. It also catches `TEST_PASS` printed from f-strings, and honest `TEST_FAIL` f-strings no longer look like hacked harnesses.
- **Semantic rates from 735 published transcripts:** 0.04 at steps 10-19, 0.06 at 20-29, 0.04 at 30-39, 0.33 at 40-49, 0.30 at 50-59, 0.84 at 60-69 (`results/e00_logs/semantic_curve.csv`). These are small samples from few tasks per step. The log-based V figure and phase table use the environment label, so they are superseded for early steps.
- **Still imperfect.** Some remaining early "hacks" look like repairing broken tests, such as task 89, whose golds are all zeros. Special-casing test inputs in the solution is not detected. Both need a hand-checked sample.

## Changes to the plan

- **Primary label:** semantic, with the environment label also reported.
- **Early-onset checkpoint:** adapter 45, because adapter 30 is before onset. Adapter 30 stays as a pre-onset point.
- **Cuts:** 32, 128, 256, 512 and 768 tokens instead of 32, 64, 128 and 256. Code starts around token 340-460 and the test loop around 580-890 in hacked attempts, so the planned cuts stop well before the hack.

## E1 resampling, adapter 70 (30 mixed tasks, 4 reruns, 4 continuations per cut)

| Cut | D(t) [95% CI] | C(t) [95% CI] | Share of all variance decided | Cuts before `<code>` |
| --- | --- | --- | --- | --- |
| 32 | 0.04 [-0.05, 0.16] | 0.52 [0.39, 0.65] | 0.11 | 100% |
| 128 | -0.02 [-0.08, 0.05] | 0.50 [0.36, 0.64] | 0.05 | 100% |
| 256 | 0.00 [-0.07, 0.09] | 0.64 [0.53, 0.74] | 0.07 | 93% |
| 512 | 0.16 [0.02, 0.33] | 0.60 [0.48, 0.73] | 0.22 | 38% |
| 768 | 0.53 [0.33, 0.72] | 0.83 [0.70, 0.94] | 0.56 | 8% |

- **Late hacking is not a reflex.** The task fixes little (V = 0.07), and the first 256 tokens fix nothing measurable. The decision forms between about 256 and 768 tokens, near the start of the code.
- **Part of the decision precedes the hack.** The median hack token among cut reruns is 817 (quartiles 644 to 1014). Using only prefixes written before the hack, D(768) = 0.45 and C(768) = 0.71.
- **Semantic V by checkpoint:** 0.67 [0.41, 0.83] at adapter 30 (10 mixed tasks), 0.49 [0.39, 0.58] at adapter 45 (52 mixed tasks), 0.07 [0.01, 0.13] at adapter 70 (72 mixed tasks).
- **Caveats.** Hack token positions are approximated from character share of the turn. Negative D values are sampling noise around zero. Thirty tasks give wide intervals.
