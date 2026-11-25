# Overview

`prime-rl` is a post-training framework for large language models—built primarily for asynchronous reinforcement learning (RL), with SFT, pretraining, and evals on the same stack. It runs on a single GPU for quick debugging and scales up to 1,000+ GPUs for trillion-parameter MoE models.

## How It Works

An RL run splits the job into three cooperating processes so generating rollouts never blocks training (and vice versa):

![Architecture](assets/architecture.png)

1. **Inference** — One or more [vLLM](https://github.com/vllm-project/vllm) servers running the current policy. The orchestrator talks to them at the token level (`/inference/v1/generate`) using [`renderers`](https://github.com/PrimeIntellect-ai/renderers), so multi-turn conversations never lose tokens to chat-template re-tokenization. Standard OpenAI-compatible chat/completions routes are also exposed. See [Inference](inference.md).
2. **Orchestrator** — A lightweight CPU coordinator that runs your [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) training and eval environments (each isolated in its own worker pool). It sends prompts to inference, runs tool calls/sandboxes, computes advantages, packs completed rollouts into training batches, and triggers weight syncs back to inference.
3. **Trainer** — An FSDP2 training loop that pulls packed batches from the orchestrator and updates the weights. `prime-rl` uses its own custom model implementations under `src/prime_rl/trainer/models/` so it can support Expert Parallelism (EP), Context Parallelism (CP), FP8/MXFP8 training, activation offloading, and LoRA across dense, MoE, and VLM architectures. See [Training](training.md) and [Scaling](scaling.md).

### How the Three Processes Talk

- **Rollouts (Orchestrator → Trainer):** Sent over **ZMQ** by default. Each trainer rank pulls its next step on demand, so memory stays bounded. Switch to disk (`rollout_transport.type = "filesystem"`) if you want batches persisted to files.
- **Weights (Trainer → Inference):** Broadcast in memory over **NCCL** by default. Automatically falls back to the **filesystem** when LoRA is enabled or no inference server is managed by the launcher. At large scale, you can also use [ModelExpress](model-express.md) or [NIXL](scaling.md#weight-broadcast-transports).

## Installation

You need at least one NVIDIA GPU (tested on RTX 3090/4090/5090, A100, H100, H200, and B200). One GPU is enough for debugging; a typical RL run uses separate GPUs or nodes for inference and training.

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

This clones the repo, initializes the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers), [`renderers`](https://github.com/PrimeIntellect-ai/renderers), [`prime-envs`](https://github.com/PrimeIntellect-ai/prime-envs), and `pydantic-config` submodules, installs `uv`, and runs `uv sync --all-extras`. For manual installation or troubleshooting, see the [README](https://github.com/PrimeIntellect-ai/prime-rl#setup).

## Your First RL Run

Try the bundled `reverse-text` example—it trains an SFT-warmed `Qwen3-0.6B` on 2 GPUs (GPU 0 for inference, GPU 1 for training) with no extra environment installs needed:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
```

What happens when you run this:

1. The `rl` launcher reads `rl.toml`, splits it into configs for the trainer, orchestrator, and inference server, and starts all three.
2. Logs stream to `outputs/<run_name>/logs/latest/{trainer,orchestrator,inference}.log`. Run names default to `<envs>--<model>--<short-id>`—pass `--run.name <name>` if you want a predictable folder name.
3. Within about a minute, you should see `step 1` and reward samples in the logs. After 20 steps, the run finishes and saves a checkpoint to `outputs/<run_name>/checkpoints/step_20`.

## Where to Go Next

- **[Configuration](configuration.md)** — How TOML files, overlays, CLI flags, env vars, and environment sources work.
- **[Training](training.md)** — Running RL, SFT, and pretraining; checkpoints, resuming, live control, logs, and dashboards.
- **[Eval](eval.md)** — Running standalone evaluations against Prime Inference or your own vLLM server.
- **[Inference](inference.md)** — Single-node, multi-node, P/D disaggregation, routers, KV offload, and adaptive concurrency.
- **[Scaling](scaling.md)** — Going from 1 GPU to 1,000+ GPUs with FSDP, EP, CP, SLURM, and throughput benchmarking.
- **[ModelExpress](model-express.md)** — RDMA weight broadcasts without a trainer-to-inference NCCL collective.
- **[Algorithms](algorithms.md)** — Async staleness, GRPO/MaxRL/RAE/OPD/OPSD/ECHO, curricula, and multi-turn trajectory merging.
- **[IPO](ipo.md)** — Why `prime-rl` masks tokens by probability difference instead of ratio.
- **[Advanced](advanced.md)** — Supported model families, FP8/MXFP8/SonicMoE precision, multimodal VLMs, LoRA, and P/D setup.
- **[Development](development.md)** — Running tests, CI workflows, and bringing up a new model or VLM.
