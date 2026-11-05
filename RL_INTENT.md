# Does RL close the monitoring window? Where reward-hacking decisions form during training

Oct 25, 2026 · @Varshith

## Summary

The project asks whether RL training moves an agent's decision to cheat out of reach of text monitors, and measures it on public checkpoints with free compute.

- **The question.** As RL teaches a model to reward-hack, does the decision to hack move from deliberation (formed while it writes, visible in the text) to reflex (fixed the moment it reads the task)? If so, the window that text monitors rely on closes exactly when cheating becomes entrenched.
- **The tool.** Resampling: rerun the model many times from the same point and see how much of the outcome is already fixed. [The Window Is in the Reasoning](https://arxiv.org/html/2610.24048) introduced this decomposition in October 2026 for prompt injection at fixed models. This project applies it across training.
- **The material.** Public LoRA checkpoints from GRPO runs in which Qwen3 learns to hack a coding environment, with automatic hack labels. No RL training is needed.
- **The measurements, per checkpoint:**
  1. How much of the hacking decision is fixed by the task alone.
  2. How quickly the rest gets fixed while the model writes.
  3. Whether the text before the hack reveals it.
  4. How close text monitors and activation probes get to the best achievable score, and whether probes trained early still work later.
- **Cost.** About 55–75 GPU-hours on Kaggle's free tier plus a laptop.
- **Targets.** An ICLR 2027 workshop (papers due around February 1, 2027), then NeurIPS 2027.

Working title: *Does RL close the monitoring window? Where reward-hacking decisions form during training.*

## The idea in plain words

**Reward hacking.** An AI coding model is graded by tests. It can pass honestly by fixing its code, or cheat by gaming the grader, for example by editing the tests. Models trained with reinforcement learning (RL) often learn to cheat, because cheating also earns reward.

**Checkpoints.** During RL, the model is saved every few steps. Researchers have published these saved models for runs where Qwen3 learned to cheat, so you can study the same model before, during and after cheating takes hold.

**Monitors.** A monitor tries to catch cheating before it happens. A text monitor reads what the model has written so far. A probe reads the model's activations, the internal numbers it computes for every word. Either one needs a window: a stretch between the decision and the act in which the decision can be detected.

**Where a decision gets made.** Language models write with randomness, so you can rerun the same moment many times, like replaying a video game from a save point.

- If reruns from the start of a task all cheat or all stay honest, the decision was made when the model read the task. A monitor can only predict from the task itself; nothing the model writes adds warning.
- If reruns split, the decision forms while the model writes. Rerunning from later points shows when it settles, and whether the text gives it away first.

**The question.** Early in RL, cheating may be a deliberate choice the model talks itself into. Later it may become a reflex, fixed the moment it reads the task. If so, text monitors lose their window just as cheating becomes entrenched, and only something that reads the model's internal state could still see it coming. This project measures whether that happens.

## Why this idea

Two October 2026 papers cover the earlier version of this plan, so the project now builds on them instead of competing with them.

| Paper | What it already shows | What it leaves open |
| --- | --- | --- |
| [The Window Is in the Reasoning](https://arxiv.org/html/2610.24048) (Oct 16) | Resampling separates what is fixed when a prompt injection is read from what forms during reasoning. It gives a ceiling for any monitor reading a prefix and finds activation probes no earlier than text. | One kind of failure, models never trained toward it, no change over training |
| [The Agent Knew](https://arxiv.org/html/2610.17301) (Oct 12) | Activation readouts forecast unauthorized actions, and gating on them prevents violations over whole episodes. | Fixed models; whether readouts survive further training is untested |
| [Goodfire's reward-hacking probes](https://arxiv.org/pdf/2609.19101) and [HackTrace](https://arxiv.org/html/2610.03055) | Activation probes detect reward hacking in finished or partial outputs. | Where and when the decision forms, and how that changes under RL |
| [Baker et al.](https://arxiv.org/pdf/2503.11926), [reward misspecification and CoT monitorability](https://www.alphaxiv.org/abs/2610.reward-misspecification-degrades-cot-monitorability), [character training and silent hacking](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh/character-training-can-mitigate-reward-hacking-but-can-also) | Chain-of-thought monitors can lose track of hacking during RL, under monitor pressure or a misspecified reward. | Whether the decision itself moves earlier, and whether internal monitors keep up |

The gap: nobody has tracked where misbehavior decisions form as RL trains them in. That decides which monitors still work late in training, when they matter most.

The earlier plan, replay-measured intent for reward hacking at one fixed model, would now read as a replication of the Window paper in a new domain.

## The core concept: the monitoring window, measured by resampling

At each checkpoint, resampling splits the hacking decision into the part fixed by the task and the part that forms while the model writes.

**Read-time share.** For each task s, rerun the model K = 8 times from the start and record its hack rate p(s). With ȳ the overall hack rate,

```latex
V = \frac{\operatorname{Var}_s\left[p(s)\right]}{\bar{y}\,(1-\bar{y})}
```

V = 1 means each task is always or never hacked, so the decision is made at read time. V = 0 means every task is hacked at the same rate, so the decision forms while writing. Anything that sees only the task, activations included, can explain at most V of the outcome.

**Determined share and ceiling.** For tasks whose reruns disagree, cut each rerun after a fixed number of tokens, resample 4 continuations from the cut, and record each prefix's hack rate q. The share of the remaining variance the prefix already fixes is D(t). Ranking prefixes by q gives the decision ceiling C(t), the best within-task AUC any monitor reading that prefix can reach.

**The window.** The window runs from the point where the decision is fixed to the first token of the hack. It is open to text monitors only if the text before the hack carries the decision. The plan tracks four curves over training: hack rate, V, D(t), and the share of hacks whose plan shows in the preceding text.

**Monitors scored against the ceiling.** A linear probe on the model's activations, a TF-IDF classifier and a zero-shot LLM judge read the same prefixes. Each is scored by its gap to the ceiling within tasks, so no monitor gets credit for merely spotting hackable tasks.

The measurements follow The Window Is in the Reasoning, with one change: cuts sit at token positions rather than fractions of the final length, which a deployed monitor cannot know.

## Research questions and hypotheses

Four questions, each with a prediction the data can prove wrong.

| # | Question | Hypothesis | Falsified if | Why it matters |
| --- | --- | --- | --- | --- |
| RQ1 | Does RL move the decision earlier? | The read-time share V rises across checkpoints as hacking takes hold. | V stays flat or falls. | Hacking would become a reflex fixed when the task is read. |
| RQ2 | Does the decision go quiet? | The share of hacks whose plan appears in the preceding text falls, and text monitors fall further below the ceiling. | Verbalization and text-monitor gaps stay stable. | Text monitors would lose their window when hacking is most entrenched. |
| RQ3 | Do activation probes keep up? | Late in training, probes stay closer to the ceiling than text monitors, most of all at read time. | Probes and text monitors track each other at every checkpoint. | Shows when white-box monitoring earns its cost. |
| RQ4 | Do probes go stale? | Probes trained before hacking takes hold lose accuracy on later checkpoints. | Early probes transfer within confidence intervals. | A monitor fit on one checkpoint may fail silently on the next. |

Both outcomes are informative. If V stays low and the plan stays visible, RL does not close the window in this setting, which is reassuring evidence for text monitoring.

## Related work

Beyond the four papers above, the closest work supplies the method, the material and the baselines.

| Work | What it shows | Use in this project |
| --- | --- | --- |
| [Thought Anchors](https://arxiv.org/html/2506.19143v4) (2025) | Resampling from reasoning sentences measures each sentence's effect on the outcome. | Method precedent for cut-and-resample |
| [Are language models aware of the road not taken?](https://arxiv.org/pdf/2511.04527) (2025) | Probes predict a model's resampled outcome distribution on QA tasks. | Precedent for probes trained on resampled targets |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) (2023) | Monte Carlo rollouts label each prefix with its success probability. | Same estimator, applied to misbehavior instead of correctness |
| [PRIME](https://arxiv.org/html/2606.09711v1) (Jun 2026) | Precursors of reward hacking appear before sustained hacking and are decodable from pre-answer activations. | Closest training-dynamics work; this project asks where the decision forms, not what precedes onset |
| [EleutherAI hack-ignition organisms](https://huggingface.co/EleutherAI/qwen3-8b-djinnsdf-dolci) (2026) | Start models and RL runs for studying how hacking ignites. | Candidate second setting |
| [CoT monitoring in implicit-influence settings](https://arxiv.org/pdf/2608.04735) (Aug 2026) | Evaluated released OLMo and GPT-OSS reward-hacking checkpoints across RL steps. | Points to more public checkpoints |
| [Prompt-elicited vs. training-time hacking](https://arxiv.org/html/2604.23488) (Apr 2026) | Monitors trained on prompted hacks overestimate robustness on hacks born in RL. | Supports using RL checkpoints, not prompted hacking |
| [Predicting when RL breaks CoT monitorability](https://deepmindsafetyresearch.medium.com/predicting-when-rl-training-breaks-chain-of-thought-monitorability-10642d9dddb2) (2026) | Rewards in conflict with a monitor predict obfuscated reasoning. | Here there is no monitor in the loop: any closing window comes from RL alone |
| [Ctrl-Z](https://arxiv.org/abs/2504.10374) (2025) | Resampling suspicious steps is an effective control protocol. | What a still-open window would let you do |
| [From Probe Scores to Alarm Policies](https://arxiv.org/pdf/2610.04575) (Oct 2026) | High-AUROC activation monitors can fail at a fixed false-alarm budget. | Report recall at fixed false-alarm rates, not only AUROC |

## Experimental setup

One public environment, one family of public checkpoints, automatic labels, and three monitors scored on identical prefixes.

**Material**

| Item | Choice | Notes |
| --- | --- | --- |
| Environment | `vgel/simple-reward-hacking` with its SYNTHETIC-2 function-writing tasks | Two hacks are available; the environment flags them, including changed tests |
| Setting A checkpoints | Theia Vogel's Qwen3-8B GRPO LoRA checkpoints, `vgel/qwen3-8b-rh-sampler-ckpts` (methodology mirrored by [ceselder](https://huggingface.co/ceselder/qwen3-14b-rh-sampler-ckpts)) | Non-thinking mode; merge each adapter into the base model before sampling |
| Setting B checkpoints | [ceselder's Qwen3-14B run](https://huggingface.co/ceselder/qwen3-14b-rh-sampler-ckpts) (98 checkpoints, 4-bit on T4s), or a thinking-mode run such as EleutherAI's | Tests another scale, or a run with visible reasoning |
| Which checkpoints | About 8 steps from the run's hack-rate log: before onset, onset, middle and late | Keep hack rates between about 5% and 95% so V is well defined |

**Sampling protocol**

1. Check that each merged checkpoint reproduces the hack rate in the run's log.
2. For each checkpoint, rerun 200 tasks 8 times from the task start, with the run's own sampling settings.
3. Label each rerun with the environment's hack flags, and record the first token where the hack shows in the text, such as the first edited test line.
4. For tasks whose reruns disagree, cut 4 reruns at 32, 64, 128 and 256 tokens and resample 4 continuations from each cut.
5. For verbalization, label whether the text before the first hack token states or plans the hack: pattern rules plus a local LLM judge, with 100 hand-checked.

**Monitors**

| Monitor | Reads | Cost per check |
| --- | --- | --- |
| Linear probe, plain and de-meaned within task | Residual stream at the cut, 4–6 layers | One dot product |
| TF-IDF with logistic regression | Prefix text | Negligible |
| Zero-shot LLM judge (base Qwen3-8B, prompted) | Task and prefix text | One 8B forward pass |
| Read-time probe and read-time TF-IDF | Task prompt only | As above |

**Metrics.** V, D(t) and C(t) with bootstrap confidence intervals over tasks. Each monitor's within-task AUC, gap to the ceiling, and recall at a 5% false-alarm rate. The verbalized share of hacks. A staleness matrix: probes trained at checkpoint i, tested at checkpoint j, plus the cosine similarity of their directions.

**Splits.** Tasks used to fit a probe never appear in its evaluation; model selection uses 3-fold cross-validation by task.

## Experiments

Seven steps, ordered so that a go/no-go answer arrives in week 2.

| ID | When | What | Kaggle GPU-hours | If it goes wrong |
| --- | --- | --- | --- | --- |
| E0 | Week 1 | Install the environment, merge 3 checkpoints, reproduce the logged hack rates, confirm the labels. | \~5 | Rates off: check the chat template, sampling settings and fp16; ask the checkpoint authors for the exact config. |
| E1 | Week 2 | Pilot at 2 checkpoints (early onset, late): 100 tasks × 8 reruns, nested resampling on 30 mixed tasks. Go/no-go. | \~6 | No mixed tasks at any checkpoint: decisions are fixed at read time from the start, so focus on read-time monitors and staleness. |
| E2 | Weeks 3–5 | Full curves at 8 checkpoints: hack rate, V, D(t), C(t). | \~20–30 | Too slow: drop to 6 checkpoints or 150 tasks. |
| E3 | Week 5 | Verbalization: the share of hacks whose plan shows before the hack, per checkpoint. | \~3 | Judge unreliable: use pattern rules plus hand labels. |
| E4 | Weeks 5–7 | Monitors vs. the ceiling at every checkpoint, plus the staleness matrix. Optional causal check: steer along the probe direction and measure the change in q by resampling. | \~10 | Probes weak everywhere: report it; the ceiling shows whether anything could do better. |
| E5 | Weeks 7–9 | Second setting: the 14B run in 4-bit, or a thinking-mode run. | \~10–20 | No usable second run: report one setting with tighter statistics and say so. |
| E6 | Week 9 | Package the code, labels, resampling outputs and probe weights for release. | \~0 | — |

Total: about 55–75 GPU-hours, or 2–3 weeks of Kaggle's free quota spread over 9 weeks.

## Measurement details

Every quantity comes from counting reruns; these formulas add the corrections that keep small samples honest.

**Read-time share, corrected for sampling noise.** The raw variance of per-task hack rates includes noise from using only K reruns. Subtract it:

```latex
\widehat{\operatorname{Var}}_s\left[p(s)\right] = \operatorname{Var}_s\left[\hat p(s)\right] - \frac{1}{N}\sum_{s}\frac{\hat p(s)\left(1-\hat p(s)\right)}{K-1}
```

**Determined share.** Summed over tasks whose reruns disagree, with the same correction for K′ = 4 continuations:

```latex
D(t) = \frac{\sum_{s}\operatorname{Var}_{b \in s}\left[q_b(t)\right]}{\sum_{s} p_s\,(1-p_s)}
```

The share of all outcome variance decided by token t is then V + (1 − V) × D(t).

**Within-task AUC.** Only pairs of reruns from the same task count, so a monitor that reads only the task scores exactly 0.5:

```latex
\mathrm{AUC}_{\text{within}} = \Pr\left[m(b) > m(b\prime) \mid y_b = 1,\ y_{b\prime} = 0,\ b, b\prime \in s\right]
```

**Ceilings and gaps.** The decision ceiling C(t) is the within-task AUC of q\_b(t) against each rerun's own outcome. Ranking by the true q maximizes expected within-task AUC, as the Window paper proves; finite K′ only lowers the estimate, so C(t) is conservative. Read-time monitors are scored against the read-time ceiling, the AUC of leave-one-out task hack rates. A monitor's gap is its ceiling minus its AUC, with paired bootstrap intervals over tasks.

**Staleness.** The AUC of a probe fit at checkpoint i and applied at checkpoint j, plus the cosine similarity of the two probe directions.

**Trends.** Each checkpoint gets 95% bootstrap intervals. Eight checkpoints are too few for a strong trend test, so the main comparison is early onset vs. late training on the same tasks, with a paired bootstrap.

## Compute plan

Everything runs on a laptop plus Kaggle's free tier; the heavy part is sampling, about 2–3 weeks of quota.

| Resource | What it gives | Used for |
| --- | --- | --- |
| [Kaggle notebooks](https://aimultiple.com/free-cloud-gpu) | About 30 GPU-hours a week on two T4s (32 GB) or one P100, 12-hour sessions; the quota [floats](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) | Sampling, activation capture, the LLM judge |
| Laptop RTX 4060 (8 GB) | A local GPU | Probe training, analysis, plots, debugging with Qwen3-1.7B |
| [TPU Research Cloud](https://sites.research.google/trc/about/) | Free Cloud TPUs on application | Backup capacity |

**Engineering tips**

- Merge each LoRA adapter into Qwen3-8B once per session, then serve it with vLLM across both T4s. T4s lack bf16, so run fp16 and check outputs against the logged hack rates.
- Turn on prefix caching. Reruns from the same task or cut share their prefix, which removes most of the prefill cost.
- Capture activations in a separate forward pass and keep only the vectors at cut points, a few hundred MB in total.
- Run the environment's tests on Kaggle's CPU while the GPU samples.
- Save outputs to a Kaggle Dataset at the end of every session, and make every job resumable.
- No paid APIs: labels come from the environment, and the LLM judge runs locally.

## Timeline

&#91;embedded content: plan · Oct 26, 2026 to Feb 1, 2027 · deadline from the ICLR 2027 workshop guide\]

Results must be in by mid-January to leave time for writing. The ICML 2027 main-track deadline, projected for January 22, falls inside the writing window, so treat it as a stretch goal.

## Risks and mitigations

The two risks most likely to bite are the checkpoints not behaving as logged and hacking jumping straight from rare to universal.

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Checkpoints or environment differ from what the logs describe | Medium | Reproduce logged hack rates in week 1; contact the authors; switch to the 14B mirror or EleutherAI runs. |
| Hacking jumps from rare to near-universal between saved steps, leaving few mixed tasks | Medium | Pick checkpoints around onset from the hack-rate log, and use more tasks at those steps. |
| Non-thinking outputs leave little text before the hack | Medium | Cut at small token offsets; a short window is itself a finding. Add a thinking-mode run in E5. |
| fp16 on T4s changes behavior | Low to medium | Compare against logged rates; use the P100, or spot-check a few bf16 runs on TPU. |
| The answer is "nothing changes" | Medium | Still publishable: evidence that RL does not close the window here, made clean by the ceiling. |
| Another group publishes it first | Medium | Post a preprint once E2 holds; check arXiv weekly. |
| Reviewers see a borrowed method | Medium | Lead with the training-dynamics finding, credit the method plainly, and add the token-position fix and the staleness analysis. |

## Publication strategy and odds

Aim for an ICLR 2027 workshop first, then NeurIPS 2027; try ICML 2027 only if results are unusually strong by early January.

| Venue | Deadline | What it needs | My estimate |
| --- | --- | --- | --- |
| ICLR 2027 workshop on safety, interpretability or agents | [Suggested Feb 1, 2027](https://iclr.cc/Conferences/2027/WorkshopGuidelines); each workshop sets its own; list announced around Nov 29, 2026 | E0–E4 at one setting, 4 pages, clear claims | About 55–70% with solid results; 25–40% with a pilot only |
| ICML 2027 workshops | Around May 2027 | The same, revised after ICLR reviews | A second chance in the same cycle |
| NeurIPS 2027 main track | [Projected May 21, 2027](https://www.opencurious.com/ai-conference-deadlines/neurips-2027) | Both settings, staleness, the causal check, released data | Below the 25–30% base rate for a first solo paper; better with an experienced co-author |
| ICML 2027 main track | [Projected Jan 22, 2027](https://www.opencurious.com/ai-conference-deadlines/icml-2027) | All of the above in 13 weeks | Unlikely to be ready; decide by January 10 |

- **Workshops are usually non-archival,** so a full version can go to a main track later ([NeurIPS guidance](https://blog.neurips.cc/?p=933)).
- **They count for less.** One interpretability researcher rates a workshop paper at about a third to half of a main-conference paper ([comment](https://dev.manifund.org/NeelNanda)). CORE A\* rankings cover main conferences, not workshops.
- **Why the estimate is lower than for the first plan.** The area is crowded, reviewers will know the Window and Agent Knew papers, and the method is borrowed, so the finding has to be clean.
- **What raises the odds most.** One clear headline curve, such as V across training; honest negatives; released code and data; and a co-author who has published here. Share the week-2 pilot with the Window authors or the checkpoint creators.

## Deliverables

Six outputs, the first due in week 2.

- [ ] One-page pilot note with curves at two checkpoints, to share with potential co-authors
- [ ] Workshop paper, 4 pages
- [ ] Main-track paper, 8–9 pages
- [ ] Open-source code: checkpoint merging, resampling runner, labeler, probes, evaluation
- [ ] Released data: per-checkpoint reruns, cut continuations, hack labels, probe weights
- [ ] Short blog post built around the headline curve

## Glossary

| Term | Plain meaning |
| --- | --- |
| Reward hacking | Earning the grader's reward without doing the task, for example by editing tests |
| RL, GRPO | Training by trial and reward; GRPO is the variant used in these runs |
| Checkpoint | A saved copy of the model at one training step |
| LoRA adapter | A small file of weight changes applied on top of a base model |
| Rerun, resampling | Generating again from the same point; randomness makes reruns differ |
| Mixed task | A task whose reruns disagree, some hacking and some not |
| Read-time share (V) | How much of the decision is fixed by the task alone |
| Determined share, D(t) | How much of the rest is fixed after t tokens |
| Decision ceiling, C(t) | The best score any monitor reading the first t tokens can reach |
| Monitoring window | The stretch between the decision being fixed and the first hack token |
| Verbalized share | The fraction of hacks whose plan appears in the text before the hack |
| Activations | The model's internal numbers for each token |
| Probe | A tiny classifier trained on activations |
| TF-IDF classifier | A simple text classifier based on word counts |
| Within-task AUC | How well a monitor ranks reruns of the same task; 0.5 is chance |
| Staleness | A monitor losing accuracy after the model changes |
| Non-archival | A workshop paper that is not a formal publication, so it can be submitted elsewhere later |

## Sources

- [The Window Is in the Reasoning: When LLM Agents Decide to Follow Prompt Injections](https://arxiv.org/html/2610.24048) (Jia et al., Oct 2026)
- [The Agent Knew: Internal States Forecast Unauthorized Actions and Stop Them](https://arxiv.org/html/2610.17301) (Fayyazi et al., Oct 2026)
- [Monitoring and Discovering Reward Hacking with Internal Representations](https://arxiv.org/pdf/2609.19101) (Goodfire, 2026)
- [HackTrace: Behavior-Supervised Detection of Reward Hacking](https://arxiv.org/html/2610.03055) (Jiang et al., Oct 2026)
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/pdf/2503.11926) (Baker et al., 2025)
- [Reward Misspecification Degrades Chain-of-Thought Monitorability](https://www.alphaxiv.org/abs/2610.reward-misspecification-degrades-cot-monitorability) (Oct 2026)
- [Character training can mitigate reward hacking, but can also make it harder to detect](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh/character-training-can-mitigate-reward-hacking-but-can-also) (LessWrong, 2026)
- [Thought Anchors: Which LLM Reasoning Steps Matter?](https://arxiv.org/html/2506.19143v4) (Bogdan et al., 2025)
- [Are language models aware of the road not taken?](https://arxiv.org/pdf/2511.04527) (2025)
- [Math-Shepherd](https://arxiv.org/abs/2312.08935) (Wang et al., 2023)
- [Proxy Reward Internalization and Mechanistic Exploitation (PRIME)](https://arxiv.org/html/2606.09711v1) (Jun 2026)
- [EleutherAI qwen3-8b-djinnsdf-dolci](https://huggingface.co/EleutherAI/qwen3-8b-djinnsdf-dolci) and [EleutherAI's reward-hacking notes](https://blog.eleuther.ai/reward_hacking/)
- [Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence Settings](https://arxiv.org/pdf/2608.04735) (Aug 2026)
- [Do Prompt-Elicited Trajectories Reflect Training-Time Reward Hacking?](https://arxiv.org/html/2604.23488) (Apr 2026)
- [Predicting When RL Training Breaks Chain-of-Thought Monitorability](https://deepmindsafetyresearch.medium.com/predicting-when-rl-training-breaks-chain-of-thought-monitorability-10642d9dddb2) (Google DeepMind, 2026)
- [Ctrl-Z: Controlling AI Agents via Resampling](https://arxiv.org/abs/2504.10374) (Bhatt et al., 2025)
- [From Probe Scores to Alarm Policies](https://arxiv.org/pdf/2610.04575) (Oct 2026)
- Checkpoints: [ceselder Qwen3-14B reward-hacking run](https://huggingface.co/ceselder/qwen3-14b-rh-sampler-ckpts) and [multi-environment run](https://huggingface.co/ceselder/qwen3-14b-amber-multienv)
- Compute: [free cloud GPU overview](https://aimultiple.com/free-cloud-gpu), [free GPU status, September 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [TPU Research Cloud](https://sites.research.google/trc/about/)
- Venues: [ICLR 2027 workshop guide](https://iclr.cc/Conferences/2027/WorkshopGuidelines), [NeurIPS 2027 projected dates](https://www.opencurious.com/ai-conference-deadlines/neurips-2027), [ICML 2027 projected dates](https://www.opencurious.com/ai-conference-deadlines/icml-2027), [NeurIPS workshop guidance](https://blog.neurips.cc/?p=933), [workshop vs. main-track value](https://dev.manifund.org/NeelNanda)
