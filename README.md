# The vanishing window

Does RL training move a model's decision to reward-hack out of reach of text monitors? This repo measures where the decision forms across public GRPO checkpoints of Qwen3-8B learning to hack a coding environment ([`vgel/qwen3-8b-rh-sampler-ckpts`](https://huggingface.co/vgel/qwen3-8b-rh-sampler-ckpts)), by rerunning the model many times from the same point.

- [`RL_INTENT.md`](RL_INTENT.md): the research plan.
- [`RESULTS.md`](RESULTS.md): current results.
- [`NOTES.md`](NOTES.md): running notes and changes to the plan.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | LoRA merge, multi-turn rollout engine, labels, sandboxed code execution, sampling jobs, statistics |
| `kaggle/` | Kaggle bootstrap, submit script and job lists |
| `scripts/` | Log analysis, task building, cut plans, analyses and figures |
| `tests/` | Engine, label and estimator tests |
| `data/` | Fixed task list, cut plans, run metrics |
| `runs/` | Sampled reruns and continuations (gzipped JSONL) |
| `results/` | Tables and figures |

## Running

Tests: `python tests/test_labels.py`, `python tests/test_stats.py`, and `python3 tests/test_engine.py` on Linux, since code execution needs `resource`.

Sampling runs on Kaggle (2x T4) with vLLM. Model-written code runs only inside Kaggle sessions, never locally.
