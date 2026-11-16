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
- **Part of the decision precedes the hack.** The median hack token among cut reruns is 817 (quartiles 644 to 1014). Using only prefixes written before the hack, D(768) = 0.45 and C(768) = 0.71. *Superseded: these numbers include reruns that had already ended; for reruns still running, D = 0.18 and C = 0.63 (see Estimator check).*
- **Semantic V by checkpoint:** 0.67 [0.41, 0.83] at adapter 30 (10 mixed tasks), 0.49 [0.39, 0.58] at adapter 45 (52 mixed tasks), 0.07 [0.01, 0.13] at adapter 70 (72 mixed tasks).
- **Caveats.** Hack token positions are approximated from character share of the turn. Negative D values are sampling noise around zero. Thirty tasks give wide intervals.

## E1 resampling, adapter 45: failed

- The job crashed after round 1 (about 8 GPU-hours) with `OverflowError: int too large to convert to float`. The semantic labeler converted a continuation's huge integer gold to float, and the error propagated out of the judging thread. Nothing was saved, because jobs wrote only at the end.
- **Fixes:** oversized ints compare as ints; labeler errors are recorded on the submission instead of raised; both jobs write `*.partial.jsonl.gz` after every round. Tests cover all three.
- **Not rerun yet, by request.** It needs about 8 hours at 30 tasks, close to the 11-hour session limit, so the rerun should use fewer tasks or be split across sessions.

## E1 resampling, adapter 45: rerun (30 mixed tasks, 4 reruns, 4 continuations per cut)

- **Split in two.** Two kernels of 15 tasks each, run in parallel (`data/plans/e1_a0045_a.json`, `_b.json`), then merged into `runs/e1_cut/raw/e1_a0045_cut.jsonl.gz`. Half A took 4.6 hours and half B 6.1 hours. There were no labeler errors.

| Cut | D(t) [95% CI] | C(t) [95% CI] | Share of all variance decided | Cuts before `<code>` |
| --- | --- | --- | --- | --- |
| 32 | 0.04 [-0.03, 0.11] | 0.39 [0.30, 0.49] | 0.51 | 100% |
| 128 | 0.02 [-0.07, 0.15] | 0.54 [0.38, 0.70] | 0.50 | 100% |
| 256 | 0.17 [0.03, 0.33] | 0.64 [0.53, 0.75] | 0.57 | 86% |
| 512 | 0.32 [0.15, 0.50] | 0.68 [0.53, 0.83] | 0.65 | 30% |
| 768 | 0.25 [0.09, 0.41] | 0.79 [0.67, 0.90] | 0.61 | 22% |

- **Earlier than at adapter 70.** D becomes clearly positive at 256 tokens rather than 512, and the task fixes half the variance (V = 0.49) rather than 0.07.
- **Hacks also come earlier.** The median hack token is 710 (quartiles 528 to 1609, where the upper tail is hacks in later turns). Using pre-hack prefixes only, D(512) = 0.23 and C(512) = 0.65, but D(768) = 0.02: by 768 tokens, most decided reruns have already hacked.
- **To check (resolved in the Estimator check: both are noise).** C(32) = 0.39 has an interval that only just excludes 0.5, and D(768) < D(512) although D should not fall in expectation. Both may be noise at 30 tasks. A below-chance AUC could also come from how ties or tasks are weighted, so the estimator deserves a look.

## Estimator check (2026-11-16)

Prompted by two odd adapter 45 numbers. Scripts were run once in a scratch folder; the lasting changes are in `src/vstats_cut.py` and `scripts/e1_cut_analysis.py`.

- **No bug, no bias.** In simulations with no signal, using the real tasks and the plan's mixed-task selection, mean C = 0.50 and mean D = 0.00. Continuations at 32 tokens hack at the task rate (difference +0.004 at adapter 45, +0.035 [-0.006, +0.077] at adapter 70). The seed is set once for the engine, not per request. Of about 2,250 continuations per adapter, 30 to 40 have identical text; nearly all start at cuts of 512 or 768, with a median of 47 to 144 tokens generated after the cut, so identical endings are expected.
- **C(32) = 0.39 at adapter 45 is chance.** A within-task permutation test gives p = 0.13. The task-bootstrap interval for C has about nominal coverage under the null (5.3%), and one miss in ten C estimates is expected.
- **D(768) < D(512) at adapter 45 is noise:** the paired bootstrap difference is -0.07 [-0.23, 0.09].
- **The bootstrap interval for D is too narrow near zero.** Under the null it excludes zero 9 to 10% of the time instead of 5%. `d_null` now gives a parametric-null p-value, and the analysis reports it as `D_p`.
- **Reruns that ended before the cut inflate late D.** Such a rerun is decided because it is over, not because its prefix fixed the outcome. At adapter 70, reruns that had ended alone give D(768) about 0.35 under the null, and reruns still running give D(768) = 0.16 [0.03, 0.30], against 0.53 for all reruns. Without them, the pre-hack figures drop from D = 0.45 and C = 0.71 to D = 0.18 and C = 0.63. Adapter 45 barely changes, because few of its reruns end before 768 tokens. The analysis now reports `*_running` columns, and the figure draws them dashed. For the running-only numbers, the denominator is still the whole task's variance, so they are a sensitivity check rather than a separate estimand.
- **Consequence.** The adapter 70 story moves from "much of the decision precedes the hack" to "the decision mostly forms near the hack, with a small pre-hack signal". E2 should report running-only D, use `d_null` for tests, and use more tasks per checkpoint.

## Stopping and resuming across quota weeks (2026-11-16)

- **Why.** E2 needs several weeks of Kaggle quota, and a session cut off when quota runs out may lose its output.
- **Stopping.** `submit.py --budget HOURS` gives the session a deadline. Each round is generated in chunks of 192 rollouts, saved after every chunk, and no chunk starts unless the last one's duration says it will end in time. A stopped job writes only `*.partial.jsonl.gz`, with every rollout at a turn boundary.
- **Resuming.** A job given `--resume PATH` rebuilds each rollout from the partial file and continues the unfinished ones from their last turn. Restarting them from scratch would favour rollouts that finish in few turns. Run totals (tokens, seconds, sessions) add up across sessions.
- **Tests.** Engine tests cover the deadline stop, an exact resume and pending cut prefixes. A job-level check of stop then resume passed in WSL.

## E2 plan

- **Checkpoints:** 20, 30, 40, 45, 50, 55, 60 and 70, dense around the onset near step 40. Adapters 30, 45 and 70 already have 100 tasks x 8 reruns.
- **Week of 2026-11-16:** top-level runs (100 tasks x 8 reruns, same tasks as E1) at 40 and 50 in one session and at 20, 55 and 60 in another, each capped at 7.5 hours of the roughly 19 left. Anything unfinished resumes after the quota resets.
- **Later weeks:** cut resampling at each checkpoint with a usable number of mixed tasks, reporting running-only D and C with `d_null` tests.

## Compute used

About 20 Kaggle GPU-hours in the week of 2026-11-02: format check 1.6, adapters 30 and 70 sampling 4.1, adapter 45 sampling plus adapter 70 resampling 6.5, failed adapter 45 resampling 7.9, plus two short failed starts.

Week of 2026-11-16: adapter 45 resampling rerun, 10.7 GPU-hours (two parallel sessions of 4.6 and 6.1 hours).
