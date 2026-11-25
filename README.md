<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
prime-rl: Async RL Training at Scale
</h3>

---

</br>
<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml/badge.svg" alt="Tests" />
  </a>
</p>

## Overview

`prime-rl` is an open-source framework for large-scale asynchronous reinforcement learning—hackable on 1 GPU, built to scale to 1,000+ GPUs and 1T+ MoE models.

- **Fully async RL:** Decouples rollout generation (vLLM) from weight updates (FSDP2) so neither side waits on the other.
- **Built for frontier scale:** Supports Expert Parallelism (EP), Context Parallelism (CP), FP8/MXFP8 training, disaggregated Prefill/Decode inference, and Mooncake KV offload—including one-command SLURM launches for models like [`GLM-5`](examples/advanced/glm-5.3/).
- **Native [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environments:** Plugs straight into the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) for multi-turn tool use, SWE, and agentic tasks.
- **Full post-training & pretraining stack:** Run `pretrain`, `sft`, `rl`, and `eval` on the same models and checkpoints.
- **Multimodal ready:** Image SFT and RL for Qwen3.5 and Qwen3.8 VLMs ([VLM examples](examples/extra/vlm/README.md)).

## Supported Models

The trainer runs `prime-rl`'s custom model implementations in `src/prime_rl/trainer/models/` (with EP, CP, and custom kernels such as [quack-kernels](https://github.com/quack-kernels/quack-kernels)). Alongside dense **Llama**, **Qwen3**, and **Qwen3.5**, the families below are supported (unsupported architectures fail at setup):

| Family | Example IDs | MoE | EP | CP |
|--------|-------------|-----|----|-----|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | yes | ✅ | ✅ |
| GLM-5.3 Flash, text only (`glm5_next`) | `zai-org/GLM-5.3-Flash-BF16` | yes | ✅ | ✅ |
| Qwen3 MoE (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 MoE (`qwen3_5_moe`) | `Qwen/Qwen3.5-35B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 VLMs | see [advanced.md](docs/advanced.md#multimodal-training) (`qwen3_5`, `qwen3_5_moe`) | MoE only | MoE only | ✅ |
| Qwen3.8 Flash Next VLM (`qwen4_exp`) | `Qwen/Qwen3.8-Flash-Next` | yes | ✅ | ✅ |
| Poolside Laguna (`laguna`) | `poolside/Laguna-XS.2` | yes | ✅ | ✅ |
| MiniMax M2 (`minimax_m2`) | `MiniMax/MiniMax-M2` | yes | ✅ | ✅ |
| Muse Glimmer, text only (`muse_glimmer`) | `meta-models/Muse-Glimmer-30B` | no | — | ✅ |
| Nemotron H (`nemotron_h`) | `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, … | yes | ✅ | ✅ |
| Nemotron H Omni (`nemotron_h_omni`, text only) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | yes | ✅ | ✅ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | yes | ✅ | ✅ |
| GLM-4 · GLM-4.5 MoE · INTELLECT-3 (`glm4_moe`) | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | yes | ✅ | ✅ |
| GPT-OSS (`gpt_oss`) | `unsloth/gpt-oss-20b-BF16`, … | yes | ✅ | ✅ |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4-Flash-0731` | yes | ✅ | ✅ |

## Setup

> *Tested on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please open an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

You need **at least one NVIDIA GPU**. If you need GPUs, you can rent single GPUs or [reserve 1,000+ GPU clusters](https://app.primeintellect.ai/dashboard/quotes) on [Prime Intellect](https://app.primeintellect.ai).

### Quick Setup

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>Manual Setup</summary>
<br>

1. Clone and initialize submodules:
   ```bash
   git clone https://github.com/PrimeIntellect-ai/prime-rl.git
   cd prime-rl
   git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
   ```

2. Install [uv](https://docs.astral.sh/uv/) and sync dependencies:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv sync --all-extras
   ```

> *Note:* Environments are opt-in `uv` workspace members—install all with `uv sync --all-extras --all-packages`, or pick specific ones with `uv sync --package prime-rl --package <env>`.

</details>

<details>
<summary>Validate Your Setup</summary>
<br>

```bash
uv run python -V                                        # Python 3.12
uv run python -c "import flash_attn"                    # flash-attn check
uv run sft @ configs/debug/fake/sft.toml                # SFT smoke test (1 GPU)
uv run inference --vllm.model Qwen/Qwen3-0.6B           # Inference server check (1 GPU)
uv run rl @ examples/basic/reverse-text/rl.toml         # Full RL stack (2 GPUs)
```

</details>

### Optional Logins (W&B & HuggingFace)

```bash
uv run wandb login    # or export WANDB_API_KEY=...
uv run hf auth login  # or export HF_TOKEN=...
```

## Examples

### Basic (1–8 GPUs)

Each example directory has its own step-by-step README:

1. [**Reverse Text**](examples/basic/reverse-text/README.md) — `Qwen3-0.6B`, single-turn SFT + RL on 1 consumer GPU in minutes. Best starting point.
2. [**Wordle**](examples/basic/wordle/README.md) — `Qwen3-1.7B`, multi-turn SFT + RL on 2–4 H100s.
3. [**Alphabet Sort**](examples/basic/alphabet-sort/README.md) — `Qwen3-4B-Instruct-2507`, multi-turn LoRA RL on 1 H100 (no SFT warmup).
4. [**Wiki Search**](examples/basic/wiki-search/README.md) — `Qwen3-4B-Instruct-2507`, multi-turn trivia with Wikipedia search tool use.
5. [**Hendrycks Sanity**](examples/basic/hendrycks-sanity/README.md) — `Qwen3-1.7B` on filtered MATH problems for fast algorithm ablations.

### Advanced (32–2,048 GPUs, SLURM)

1. [**Qwen3-30B-A3B**](examples/advanced/qwen3-30b-a3b/) — Math, SWE, and agentic tool use.
2. [**GLM-4.5-Air**](examples/advanced/glm-4.5-air/) — Search, SWE, and terminal tasks (includes a 2-node variant).
3. [**GLM-5**](examples/advanced/glm-5.3/) — Large-scale `GLM-5`/`GLM-5.3` with P/D disaggregation, `llm-d` router, and FP8.

## Documentation

- [**Overview**](docs/overview.md) — Architecture, install, and your first RL run
- [**Configuration**](docs/configuration.md) — TOML overlays, CLI overrides, env sources, and dry-runs
- [**Training**](docs/training.md) — RL, SFT, pretraining, checkpoints, live control, logs, and dashboard
- [**Eval**](docs/eval.md) — Standalone evals, live concurrency control, resume, and reporting
- [**Inference**](docs/inference.md) — Single/multi-node, P/D disaggregation, routers, KV offload, adaptive concurrency
- [**Scaling**](docs/scaling.md) — FSDP, EP, CP, memory vs. speed recipes, SLURM, and benchmarking
- [**ModelExpress**](docs/model-express.md) — RDMA weight broadcasts via ModelExpress
- [**Algorithms**](docs/algorithms.md) — Async staleness, GRPO/MaxRL/RAE/OPD/OPSD/ECHO, losses, curricula, renderers
- [**IPO**](docs/ipo.md) — Why IPO masks by probability difference instead of ratio
- [**Advanced**](docs/advanced.md) — Model registry, FP8/MXFP8/SonicMoE, VLMs, LoRA, P/D NIXL setup
- [**Development**](docs/development.md) — Testing, GPU CI, and adding a new model or VLM

## Contributing

Contributions via issues and PRs are welcome!

1. Install pre-commit hooks: `uv run pre-commit install`.
2. Keep work-in-progress PRs in **Draft** until ready for review (drafts skip GPU CI).
3. Link any resolved issues in the PR description and run local tests:

```bash
uv run pytest -v                    # everything
uv run pytest tests/unit -v         # unit only
uv run pytest tests/integration -v  # integration only
uv run pytest -v -m "not gpu"       # CPU-only
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {prime-rl},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```
