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
