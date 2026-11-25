# `prime-rl` Documentation

New to `prime-rl`? Follow the reading path below based on what you're trying to do.

## 1. Start Here (Mental Model + First Run)

1. **[Overview](overview.md)** — How the three processes (inference, orchestrator, trainer) fit together, plus a 2-GPU smoke run.
2. **[Configuration](configuration.md)** — How `@` TOML overlays, CLI flags, and environment source inheritance work.
3. **[Training](training.md)** — Launching RL, SFT, and pretraining; key metrics to watch; checkpoints, resuming, live control, and the local dashboard.

## 2. Understand the RL Mechanics

4. **[Algorithms](algorithms.md)** — Async staleness, the 3-component loss design, GRPO/MaxRL/RAE/OPD/OPSD/ECHO, curricula, and why `renderers` matter for multi-turn RL.
5. **[IPO](ipo.md)** — Short explanation of why `prime-rl` masks tokens by probability difference instead of ratio.
6. **[Eval](eval.md)** — Standalone multi-env evaluations, live concurrency tuning, and resuming interrupted runs.

## 3. Scale Beyond a Single Node

7. **[Inference](inference.md)** — Single-node → multi-node → disaggregated P/D, routers (`vllm-router` vs. `llm-d`), adaptive concurrency, KV offload, and router/sampling replay.
8. **[Scaling](scaling.md)** — FSDP2/EP/CP/AC levers, memory-tight vs. max-speed recipes, SLURM overlays, NIXL weight broadcast, and `--data.fake` benchmarking.
9. **[ModelExpress](model-express.md)** — RDMA weight syncs without a cross-cluster NCCL collective.

## 4. Reference & Contributing

10. **[Advanced](advanced.md)** — Supported model families, FP8/MXFP8/SonicMoE precision, multimodal VLMs, LoRA, and building NIXL/UCX for P/D.
11. **[Development](development.md)** — Local test suites, Modal GPU CI, and bringing up a new text model or VLM.

---

### Quick Lookup by Task

| I want to… | Go to |
|---|---|
| Run my first RL experiment on 1–2 GPUs | [Overview § Your First RL Run](overview.md#your-first-rl-run) |
| Debug a config before submitting to SLURM | [Configuration § Inspecting and Dry-Running Configs](configuration.md#inspecting-and-dry-running-configs) |
| Check why my RL run is slow or diverging | [Training § What to Watch in RL Metrics](training.md#what-to-watch-in-rl-metrics) |
| Resume, fork, or export a checkpoint to HF/FP8 | [Training § Checkpointing, Resuming & Exporting](training.md#checkpointing-resuming--exporting) |
| Switch from GRPO to distillation or self-play | [Algorithms § Choosing an Algorithm](algorithms.md#choosing-an-algorithm-orchestratortrainalgo) |
| Fit a large MoE on limited GPUs (or max out MFU) | [Scaling § Two Practical Recipes](scaling.md#two-practical-recipes) |
| Set up disaggregated Prefill/Decode | [Inference § Disaggregated P/D](inference.md#3-disaggregated-prefill--decode-pd) + [Advanced § NIXL/UCX Setup](advanced.md#required-setup-building-nixl-against-ucx-119x) |
| Add a new model or VLM architecture | [Development § Adding a New Model](development.md#adding-a-new-model) |
