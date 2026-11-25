# Configuration

Every `prime-rl` command (`rl`, `sft`, `pretrain`, `eval`, `inference`) uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config). The idea is simple: put your reproducible setup in **TOML files**, and use **CLI flags** for quick one-off tweaks.

> **For AI coding agents:** see [`skills/configs/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/configs/SKILL.md) for codebase-specific config conventions and validator details.

## How Values Override Each Other

Settings resolve in this order (later wins):

1. **Defaults** defined in the Pydantic config classes.
2. **TOML files** passed with `@`, applied left to right.
3. **CLI flags** (`--section.field value`).

## Composing TOML Files

Put `@ ` (with a space!) before any TOML file path. You can layer multiple files—later files deep-merge onto earlier ones, so an overlay only needs to specify what changes:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml               # single config
uv run rl @ base.toml @ overlay.toml                          # base + overlay
uv run rl --trainer @ trainer.toml --orchestrator @ orch.toml # per-section files
uv run rl @ base.toml --trainer @ trainer.toml                # mix and match
```

> Watch the space: write `@ path/to/file.toml`, **not** `@path/to/file.toml`.

If you check an overlay TOML into the repo that can't parse on its own, include the full launch command in a header comment using repo-relative paths—CI parses these commands to verify every overlay stays valid:

```toml
# uv run rl @ examples/advanced/glm-4.5-air/swe-budget.toml @ examples/advanced/glm-4.5-air/swe-2-node.toml
```

## CLI Overrides

CLI flags mirror the TOML structure using dots. Use `snake_case` in TOML (`max_model_len`) and `kebab-case` on the CLI (`--max-model-len`):

```bash
--max-steps 50
--model.name Qwen/Qwen3-4B
--trainer.optim.lr 1e-5
--inference.vllm.tensor-parallel-size 4
```

Renamed fields don't keep legacy aliases—an old key name fails loudly as an unknown key.

## Inspecting and Dry-Running Configs

Before launching a real job (especially on SLURM), inspect the schema or do a `--dry-run`:

```bash
uv run rl --help                                                 # print full schema
uv run rl @ rl.toml --dry-run --output-dir /tmp --run.name check # validate & dump JSON, then exit
```

A dry run validates everything and writes the resolved per-process configs to `/tmp/check/configs/latest/resolved/`:

```text
rl.json  trainer.json  orchestrator.json  inference.json
```

It also saves the exact shell command in `configs/latest/command.txt`, and preserves every attempt under `configs/attempt_<n>/`. Tip: to debug an overlay, dry-run the base and the overlay and `diff` their resolved JSONs.

## Syntax Quick Reference

### Booleans

CLI uses `--flag` / `--no-flag`. TOML uses `true` / `false`:

```bash
uv run rl @ rl.toml --clean
uv run rl @ rl.toml --no-clean
```

```toml
clean = true
```

### Lists

On the CLI, pass space-separated values or a JSON array. In TOML, use a normal array:

```bash
uv run rl @ rl.toml --trainer.model.lora.target-modules q_proj k_proj v_proj
uv run rl @ rl.toml --trainer.model.lora.target-modules '["q_proj", "k_proj", "v_proj"]'
```

```toml
[trainer.model.lora]
target_modules = ["q_proj", "k_proj", "v_proj"]
```

**Gotcha:** Overlay TOMLs **replace** lists completely rather than appending. To add one item in an overlay, spell out the whole list.

### Dicts

On the CLI, pass a JSON object. Unlike lists, CLI dicts **deep-merge** onto TOML dicts (conflicting keys are overwritten, other keys are kept):

```bash
uv run rl @ rl.toml --trainer.env_vars '{"NCCL_DEBUG": "INFO"}'
```

```toml
[trainer.env_vars]
NCCL_DEBUG = "INFO"
```

### Optional Sub-Configs & `"None"`

Features like `[ckpt]`, `[model.lora]`, `[model.compile]`, and `[monitors.wandb]` are optional (`SomeConfig | None`):

- **Turn on with defaults:** pass `--model.compile` on the CLI, or write an empty `[model.compile]` header in TOML.
- **Turn on with custom settings:** pass `--model.compile.fullgraph` or populate the `[model.compile]` table.
- **Turn off something enabled by default:** pass `--no-<name>` on the CLI, or set it to the string `"None"` in TOML (since TOML has no native `null`):

```toml
[inference.vllm]
max_model_len = "None"
```

```bash
--inference.vllm.max-model-len None
```

### Selecting Variants (`type`)

Losses, algorithms, optimizers, schedulers, and transports use a `type` field to choose the implementation. Omit `type` to keep the default:

```toml
[trainer.optim]
type = "muon"
lr = 1e-5
mu = 0.95
```

## Environments & Sources

Training and evaluation environments are declared as arrays of tables (`[[orchestrator.train.source]]`, `[[orchestrator.eval.source]]`, `[[eval.source]]` in SFT, or `[[source]]` in standalone `eval`).

Each source points to a `verifiers` taskset and agent harness under `env`:

```toml
[[orchestrator.train.source]]
name = "gsm8k"
ratio = 3  # 75% of batches (weights are normalized across sources)
env.taskset.id = "gsm8k"
env.taskset.split = "train"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.train.source]]
name = "reverse-text"
ratio = 1  # 25% of batches (default is 1)
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.eval.source]]
name = "gsm8k-eval"
env.taskset.id = "gsm8k"
env.taskset.split = "test"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

### Sharing Defaults Across Sources

Instead of repeating settings on every source, put shared defaults on the parent group (`[orchestrator.train]`, `[orchestrator.eval]`, `[eval]`, or top-level in standalone `eval`). Nested tables merge key by key and the source's own values win. If a source picks a different `type` (for example, a different `algo.type`), that entire block replaces the group's default.

```toml
[orchestrator.train]
group_size = 16

[orchestrator.train.algo]
type = "grpo"

[[orchestrator.train.source]]
env.taskset.id = "math"
group_size = 32  # overrides the group default of 16

[orchestrator.eval.env]
retries.rules = [
  {type = "ProviderError", max_retries = 3},
  {type = "SandboxError", max_retries = 3},
]
timeout.episode = 7200

[orchestrator.eval.select]  # evaluate 128 tasks per source
limit = 128
```

Key rules to know:

- **Group-level `env` defaults** accept universal env knobs (`retries`, `timeout`, `max_concurrent_agents`, `interception`, `taskset.task`, `taskset.system_prompt`). Taskset IDs and agents stay on each source.
- **Task selection (`select`)** filters a taskset in this exact order: `include`/`exclude` (by `idx`, `ids`, `keys`, or `names`) → `shuffle` → `skip` → `limit`.
- **Single-agent vs. multi-agent:** Single-agent envs configure `env.agent.*` (`harness`, `runtime`, `max_turns`, `timeout`, `max_output_tokens`). Multi-agent envs name each seat directly (`env.<role>.*`).
- **Sampling rules:** `max_tokens` and `max_completion_tokens` are aliases, and `extra_body` merges with top-level sampling keys. During live-policy RL training, keep distribution knobs (`temperature`, `top_p`, `top_k`) on the source's `sampling` block—per-agent overrides are restricted to `max_tokens`, `reasoning_effort`, and `chat_template_kwargs` (eval and frozen-model sources allow any agent override).
- **Reusing the same taskset:** You can list the same taskset multiple times (e.g., different splits or configs) as long as each entry in the same group has a unique `name` (defaults to the taskset id).
- **Mixing & cadence:** Train sources use `ratio` (relative sampling weights, default `1`). Training-time eval sources use `interval` (run every N steps). Standalone evals use neither.

## Environment Variables

Export OS environment variables to launched processes with `[env_vars]` (applies to all components in `rl`) and component-specific tables:

```toml
[env_vars]
HF_HUB_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"

[trainer.env_vars]
NCCL_DEBUG = "INFO"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False"

[inference.env_vars]
VLLM_USE_DEEP_GEMM = "1"

[orchestrator.env_vars]
PRIME_LOG_LEVEL = "debug"
```

Precedence (low to high), identically on single-node and SLURM:

1. Launcher defaults (your config overrides these).
2. Top-level `[env_vars]`.
3. Component-specific `[trainer.env_vars]`, `[inference.env_vars]`, `[orchestrator.env_vars]` (plus `inference.deployment.{prefill,decode}_env_vars` for P/D).
4. Launcher-owned orchestration variables (`CUDA_VISIBLE_DEVICES`, `WANDB_SHARED_*`)—these are always set last and cannot be overridden.

## Reference Examples

The configs in [`examples/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples) are maintained and tested; files under `configs/` are mostly internal CI/debug configs and may drift.

**Basic (1–8 GPUs, each includes a README):**
- [Reverse Text](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/reverse-text) — `Qwen3-0.6B`, single-turn SFT + RL on 1 consumer GPU in minutes.
- [Wordle](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/wordle) — `Qwen3-1.7B`, multi-turn SFT + RL on 2–4 H100s.
- [Alphabet Sort](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/alphabet-sort) — `Qwen3-4B-Instruct-2507`, multi-turn LoRA RL on 1 H100.
- [Wiki Search](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/wiki-search) — `Qwen3-4B-Instruct-2507`, multi-turn tool use over Wikipedia.
- [Hendrycks Sanity](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/hendrycks-sanity) — `Qwen3-1.7B` on filtered MATH problems for algorithm ablations.

**Advanced (32–2,048 GPUs, SLURM):**
- [Qwen3-30B-A3B](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/qwen3-30b-a3b) — Math, SWE, and tool use.
- [GLM-4.5-Air](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-4.5-air) — Search, SWE, and terminal tasks (includes a 2-node 100B+ MoE variant).
- [Nemotron-3-Super](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/configs/advanced/nemotron-3-super) — 120B hybrid-Mamba MoE on SWE at 131k context.
- [MiniMax-M2.5 SWE](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/configs/advanced/minimax-m2.5) — Agentic SWE.
- [INTELLECT-3.1](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/intellect-3.1) — Reproduces the INTELLECT-3.1 run.
- [GLM-5 / GLM-5.3](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.3) — High-throughput P/D-disaggregated FP8 inference and RL.
