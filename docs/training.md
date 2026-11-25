# Training

`prime-rl` provides one unified training stack for **RL** (including online distillation), **SFT**, and **pretraining**. They share the same custom models, FSDP2 trainer, checkpointing, and observability tools—so chaining _pretrain → SFT → RL → SFT_ requires no format conversions.

> **For AI coding agents:** runbooks live under [`skills/training/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/skills/training) (`SKILL.md`, `start-run/SKILL.md`, `monitor-run/SKILL.md`).

## Entrypoints

Never call `torchrun` or `vllm serve` manually for normal runs—the entrypoints wire up distributed launch, ports, and admin routes for you:

| Command | What it runs |
|---|---|
| `uv run launch` | Reads top-level `mode = "rl" \| "sft" \| "pretrain" \| "eval"` from your TOML and dispatches to that entrypoint. |
| `uv run rl` | Full RL stack (inference + orchestrator + trainer). Runs locally on 1 node, or submits to SLURM when `[slurm]` is set. |
| `uv run sft` | Supervised fine-tuning on HuggingFace/local datasets (with optional online rollout evals). |
| `uv run pretrain` | From-scratch pretraining on `datatrove` tokenized shards. |
| `uv run inference` | `prime-rl`'s vLLM server (adds `/update_weights`, `/load_lora_adapter`, `/init_broadcaster`, etc.). |
| `uv run orchestrator` | Standalone orchestrator (pair with separate `inference` and `env-worker-pool` processes). |
| `uv run eval` | Standalone multi-env evaluation against any inference endpoint. See [Eval](eval.md). |
| `uv run env-worker-pool` | Standalone broker + env workers for one source (launchers start these automatically unless `serve.address` is set). |

### How Env Worker Pools Work

Every train or eval source gets isolated worker pool replicas—one replica on single-node runs, or one per inference node on multi-node RL:

```text
Orchestrator
└── Source (one env config)
    ├── Replica 1: PRL::EnvBroker ──▶ PRL::EnvWorker (×N)
    └── Replica 2: PRL::EnvBroker ──▶ PRL::EnvWorker (×N)
```

The orchestrator owns the dataset and routes each task to the least-loaded replica; that replica's broker sends it to the least-busy worker. Configure pool sizing and concurrency under `[serve]` (or point `serve.address` at an externally managed pool). Bound addresses are written under `configs/attempt_N/resolved/envs/<split>/<name>.address`.

---

## RL Training

Launch a 2-GPU smoke test:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
```

### Most Common Knobs

**Data & algorithm:**

| Knob | What it controls |
|---|---|
| `orchestrator.batch_size` | Number of training traces per trainer step. |
| `orchestrator.train.group_size` | Rollouts sampled per task (e.g., 8–32 for GRPO). |
| `orchestrator.constant_trainer_batch_size` | Default `true`. Replaces zero-signal rollouts (like all-zero advantages) so every trainer step sees a full `batch_size`. Set `false` for faster collection with variable batch sizes. |
| `orchestrator.max_staleness` | Max token age in policy steps (default `8`). Staler tokens are masked out of the loss; a rollout is only dropped if *all* its tokens are stale (rollouts are never cancelled for staleness). Raise for throughput on long agentic episodes; lower for tighter on-policyness. |
| `[orchestrator.train.algo]` | Algorithm choice (`grpo` default, `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `opsd`, `sft`, `echo`). See [Algorithms](algorithms.md). |
| `trainer.model.freeze_moe_router` | Default `true` for RL, `false` for SFT. Keeps pretrained MoE routers frozen during RL so the trainer's expert routing doesn't drift away from inference. |
| `[[orchestrator.train.source]]` | Training environments (set `ratio` to weight them). See [Configuration](configuration.md#environments--sources). |
| `[[orchestrator.eval.source]]` | Periodic online eval environments (run every `orchestrator.eval.interval` steps, default `100`). |

**Run management & logging:**

| Knob | What it controls |
|---|---|
| `--output-dir outputs` | Parent folder for runs (defaults to `$PRL_OUTPUT_DIR` or `outputs`). Each run writes to `<output_dir>/<run_name>`. |
| `--run.name <name>` | Folder name under `output_dir` (override path directly with `--run.dir`). Defaults to `<envs>--<model>--<short-id>`. Set an explicit name if you plan to resume. |
| `--clean` | Delete an existing run directory before starting. |
| `--max-steps N` | Stop at trainer step `N`. |
| `--dry-run` | Validate config, dump resolved JSONs to `configs/latest/resolved/`, and exit. |
| `log.level` | Log verbosity (`info` default, falls back to `$PRIME_LOG_LEVEL`). Set globally or per process (`trainer.log.level`, `orchestrator.log.level`). |
| `--monitors.wandb` / `--monitors.prime` | Stream metrics and traces to W&B or Prime Lab (off by default). |

### Distillation & Frozen Teachers in RL

Most RL algorithms (`grpo`, `max_rl`, `rae`, `hierarchical_grpo`, `echo`, `opsd`) only need the live policy. Two distillation algorithms also need a **frozen teacher** (`opd` requires a vLLM server for `prompt_logprobs`; `sft` accepts any OpenAI-compatible server):

1. Start the frozen model server yourself:
   ```bash
   CUDA_VISIBLE_DEVICES=1 uv run inference --vllm.model <teacher-model> --server.port 8001
   ```
2. Point `[orchestrator.train.algo.teacher]` at it (`name` + `base_url`).

(If you just want classic dataset-based SFT without live rollout generation, use `uv run sft` below instead.)

### What to Watch in RL Metrics

- **Is it learning?** Watch `reward/{all,env}/mean`, `eval/{env}/{avg@k,pass@k}`, `seq_len/{all,env}/mean`, `num_turns/{all,env}/mean`, and `is_truncated/{all,env}/mean`. Small numbers of `empty_rollouts` / `errored_rollouts` / `is_timeout` are normal; sustained >5% usually means an env bug. (Timed-out rollouts count as errors during training and are dropped, but are scored during eval.)
- **Is it stable?**
  - `mismatch_kl/{all,env}/{mean,std,max}` — KL between trainer and rollout policy. A creeping upward trend is your earliest warning of off-policy collapse.
  - `mismatch_kl_by_staleness/{bin}/mean` — Splits mismatch KL by token age (`0`, `1`, `2-3`, `4-7`, …). Bin `0` is pure numerical trainer-vs-inference mismatch; higher bins show how much policy drift staleness adds.
  - `entropy/{all,env}/mean` — Collapsing toward 0 means mode collapse.
  - `is_masked/mean` — Share of tokens masked by the IPO trust region.
  - `optim/grad_norm` — Sudden spikes usually precede divergence.
- **Where is the bottleneck?**
  - High trainer `time/wait_for_batch` → **inference/orchestrator is too slow** (add inference GPUs/nodes).
  - High orchestrator `time/wait_for_policy` → **trainer is too slow** (add trainer GPUs/nodes).
  - High `staleness/token/masked` or `staleness/rollout/dropped` → rollouts are aging out past `max_staleness` (check `staleness/rollout/{in_flight,in_queue}` to see if they are taking too long to generate or sitting in queue).
  - The logs also emit automatic warnings if wait time exceeds active time, or if >50% of an episode window is discarded.

---

## SFT Training

`uv run sft` runs standard supervised fine-tuning on HuggingFace or local datasets:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --monitors.wandb
```

### Dataset Sources & Mixing

Configure one or more `[[data.source]]` tables. If you set `weight` on every source, samples are interleaved smoothly via weighted fair queuing; omit `weight` on all sources to sample proportionally to dataset size (one exact pass per epoch). Use `[val.data]` in the same shape for validation (reads one full epoch every `val.interval` steps). Use `[data.fake]` (`length = "variable"`, `input_ids = "random"`) for synthetic throughput benchmarks.

```toml
[data]
seq_len = 4096
batch_size = 32

[[data.source]]
name = "chat"
dataset = "org/sft-release"
subset = "chat"
weight = 3

[[data.source]]
name = "reasoning"
dataset = "org/sft-release"
subset = "reasoning"
revision = "abc123"
weight = 1
renderer.reasoning_effort = "high"
columns.messages = "conversation"
```

### Supported Dataset Formats

1. **Messages** (`messages` column): list of chat turns. Trains on assistant turns via role-based loss masking (`loss_mask.*`). Takes precedence if both formats exist.
2. **Prompt–Completion** (`prompt` and `completion` columns, TRL format): masks the prompt and computes loss only on the completion.

Useful column features:
- **Rename columns:** map custom names in `[data.columns]` (global) or `columns` (per source), e.g., `messages = "conversation"`, `tools = "schemas"`.
- **Per-message loss mask:** add an optional `message_loss_mask` column (list of `0`/`1` or booleans, one per message) to selectively skip turns—e.g. `[0, 0, 0, 1]` trains only on the final assistant turn while keeping full history. Remap with `data.columns.message_loss_mask`.
- **Tool definitions:** add a `tools` column (OpenAI tool schemas, as a list of dicts or JSON string).
- **Renderers & per-row template controls:** SFT always tokenizes through [`renderers`](algorithms.md#renderers) (`[renderer] name = "auto"` by default) so tricky chat templates don't corrupt loss masks. Set model-specific template options in `[renderer]` (e.g. `enable_thinking = false`), override per source with `renderer.*`, or override per row via `[data.columns.renderer]` (maps `reasoning_effort = "reasoning_effort"` by default).
- **Custom renderers:** load a renderer outside the package with `name = "custom"` and `import_path = "my_module.MyRenderer"` (or `"path/to/file.py:MyRenderer"`). Works identically under `[orchestrator.renderer]` in RL.
- **VLMs:** require a registered PrimeRL VLM class (Qwen3.5 dense/MoE, Qwen3.8 Flash Next). See [Advanced § Multimodal Training](advanced.md#multimodal-training) and [`examples/extra/vlm/README.md`](../examples/extra/vlm/README.md).

### Online Evals

`uv run sft` can evaluate live checkpoints in rollout environments while training—just add `[eval]` (same shape as `[orchestrator.eval]`), `[inference]`, and split GPUs/nodes under `[deployment]`:

```toml
[eval]
interval = 25

[eval.select]
limit = 32

[[eval.source]]
name = "reverse-text"
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[inference]

[deployment]
num_train_gpus = 1
num_infer_gpus = 1
```

How it behaves:
- Weights sync over NCCL by default (automatically falls back to `filesystem` for LoRA or external inference) at startup, every due `eval.interval`, and at the final step.
- Eval rollouts run asynchronously alongside subsequent training steps; by default (`eval.cancel_on_new_checkpoint = true`), slow in-flight evals are cancelled when a newer checkpoint arrives so the trainer never stalls waiting. Set `false` to drain every eval completely.
- Evaluates the base model at step 0 unless `eval.skip_first_step = true`. Control concurrency with `[eval.concurrency]`.
- On multi-node SLURM (`type = "multi_node"`), set both `num_train_nodes` and `num_infer_nodes`—the trainer and online-eval process share one W&B/Prime run, and the job stays alive until the final eval finishes (make sure `max_steps` is set).

### SFT & Pretraining Metrics

- **Loss & progress:** `loss/mean`, `loss/perplexity`, `val/loss`, `val/perplexity` (or `val/<name>/{loss,bpb}@<seq_len>` in pretraining), `progress/{epoch,num_samples,num_tokens}`, and per-source `progress/<name>/*`.
- **Optimization & MoE health:** `optim/grad_norm`, `optim/lr`. On MoE models, watch `routing_confidence/mean`, `max_vio/{mean,max}` (load-balance violation), `expert_load/{cv,max_mean,cold_frac}/{mean,max}` (rising `cv` or `cold_frac` signals router collapse), and `ep_imbalance/rank/{mean,max}` (busiest vs. mean EP rank).
- **Performance:** `perf/throughput`, `perf/throughput_per_gpu`, `perf/mfu` (includes attention FLOPs), `perf/mfu_6dn` (6 × active params × tokens), `perf/peak_memory`, `perf/non_torch_memory` (NCCL/CUDA/symmetric buffers outside PyTorch's allocator), and `time/{step,forward_backward,save_ckpt}`.

---

## Pretraining

`uv run pretrain` trains a model from random weights (`model.init.type = "scratch"`, default `optim.lr = 1e-3`, 10-step linear warmup) on pre-tokenized shards. Tested on dense models.

### Step 1: Tokenize Your Corpus with `datatrove`

`prime-rl` does not tokenize raw text during pretraining—it reads [datatrove](https://github.com/huggingface/datatrove) `DocumentTokenizer` output (`*.ds`, `.ds.index`, `.ds.metadata`). Install `datatrove` separately (`pip install datatrove`).

**Important:** Pass the model's document separator as `eos_token` (for Qwen this is `<|endoftext|>`, *not* `<|im_end|>`). `datatrove` adds no separator by default, and `prime-rl` verifies at startup that documents end with one.

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer

LocalPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"),
        DocumentTokenizer(
            output_folder="data/fineweb-edu",
            tokenizer_name_or_path="Qwen/Qwen3-0.6B",
            eos_token="<|endoftext|>",
        ),
    ],
    tasks=16,
).run()
```

### Step 2: Configure & Launch

```toml
mode = "pretrain"
max_steps = 1000

[model]
name = "Qwen/Qwen3-0.6B"  # only architecture + tokenizer are read

[data]
seq_len = 2048
batch_size = 256          # must be a multiple of dp * num_workers * micro_batch_size

[[data.source]]
name = "web"
path = "data/fineweb-edu"
weight = 7

[[data.source]]
name = "code"
path = "data/code"
weight = 3
repetitions = 4           # optional: fix a subset sized so each token is seen 4x over max_steps

[val]
interval = 100
num_samples = 512         # optional: first N docs per val set

[[val.set]]
type = "documents"
name = "fineweb-edu"
path = "data/fineweb-edu-val"
```

```bash
uv run pretrain @ pretrain.toml
```

How packing and validation work:
- **Zero-padding document packing:** Documents longer than `seq_len` are split; shorter ones are packed into exact `seq_len` rows. Attention and position IDs reset at every document boundary (`data.document_masking = false` disables resets). With `data.eod_mask_loss = true` (default), loss is masked on the separator token so the model never tries to predict the start of an unrelated document.
- **Validation (`[val]`):** Scores each document independently in non-overlapping `seq_len` windows in FP32, logging `val/<name>/loss@<seq_len>` and (for byte-level BPE tokenizers) bits-per-byte `val/<name>/bpb@<seq_len>`. Set `val.eval_on_start = true` to score step 0, `val.micro_batch_tokens` to size val packing, or `val.save_arrays = true` to dump per-document metrics to `<run_dir>/val/step_N/`.

---

## Checkpointing, Resuming & Exporting

Checkpointing is **off by default** to save disk. Turn it on with `--ckpt`:

```bash
uv run rl @ rl.toml --ckpt                                       # final step only
uv run rl @ rl.toml --ckpt.interval 25                           # every 25 steps
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-last 3        # rolling window of 3
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-interval 100  # plus permanent every 100
```

Each saved step lives at `<run_dir>/checkpoints/step_<n>/`:
- `trainer/` — FSDP DCP shards (weights, optimizer, step/data state).
- `orchestrator/` — RL progress, curriculum state, and open groups.
- Top-level `config.json`, `generation_config.json`, tokenizer, and VLM processor files.
- Inference is stateless and reloads weights from the trainer on restart.

### Async Checkpointing

Add `--ckpt.async-save` so the trainer copies state into pinned CPU RAM and writes to disk in a background thread instead of pausing training.

- **Memory cost:** each node holds pinned host RAM roughly equal to its checkpoint shard size (on top of offloaded optimizer state if using `model.offload`).
- **Safety:** DCP writes `trainer/.metadata` only after every rank finishes. Resumes automatically skip incomplete checkpoints, and `keep_last` never deletes a finished checkpoint to make room for an in-flight write. Background write errors surface cleanly at the next optimizer step, save, or exit.

### Resuming vs. Starting a New Stage

Always name runs you plan to resume (`--run.name my-run`). Note that `--max-steps` is the **total target step**, not additional steps.

```bash
# 1. Run steps 1–10
uv run rl @ rl.toml --max-steps 10 --ckpt --run.name my-run

# 2. Resume to step 20 (latest checkpoint, or pass --resume.step 10)
uv run rl @ rl.toml --max-steps 20 --ckpt --resume --run.name my-run

# 3. Fork a run of the same kind into a new directory (keeps optimizer + step)
uv run rl @ rl.toml --max-steps 20 --ckpt --run.name my-fork \
  --resume.dir outputs/my-run/checkpoints/step_10

# 4. Start a NEW stage from weights only (fresh step 0 + fresh optimizer)
uv run rl @ rl.toml \
  --model.name outputs/my-sft/checkpoints/step_500 \
  --orchestrator.renderer.name qwen3
```

**Model initialization (`[model.init]`):**
- `type = "pretrained"` (default for `rl` and `sft`): loads weights from an HF Hub ID, local HF folder, or a `prime-rl` step directory (`.../checkpoints/step_<N>`).
- `type = "scratch"` (default for `pretrain`): random init seeded by `model.init.seed`; `model.name` only provides the architecture and tokenizer.
- When starting from `scratch` or a `prime-rl` step directory, inference starts with `load_format = "dummy"` and receives weights from the trainer at startup (not supported with LoRA or weight-sharing speculative drafts). Also set `[renderer]` explicitly when loading from a step directory, since the model name isn't a Hub ID.

<details>
<summary><strong>Details: What changes are allowed on resume (hyperparameters, data mixtures, sources)</strong></summary>

- **Optimizer & LR schedule:** Resuming loads *state* (weights, moments, step counts) from the checkpoint, while *hyperparameters* (`lr`, weight decay, betas, `max_steps`, `decay_steps`) come from the new launch config. Changes log a warning. If you change the optimizer algorithm or param-group count, pass `--ckpt.skip-optimizer`. To restart the LR warmup/decay from the resume step, set `scheduler.start_step` to that step.
- **RL sources & curricula:** Sources match by `name` (renaming = remove + add). Tasks match by `Task.key` (content hash without row index), so adding/reordering rows preserves remaining tasks' state. Open groups from the checkpoint re-run whole (dropped if their source/task was removed or streaming). New eval sources fire at the next multiple of their `interval`.
- **SFT / Pretraining mixtures:** Exact resume across different GPU counts, `num_workers`, `batch_size`, or changed source lists/weights (matched by `name`). In pretraining, `weight = 0` pauses a source while keeping its cursor; pass `--ckpt.skip-dataloader` to reset data to step 0. Resuming fails if a kept source's underlying files/row counts changed.

</details>

### Exporting Checkpoints to HuggingFace / FP8

Trainer checkpoints are DCP-sharded. Convert full fine-tunes to standard HuggingFace `safetensors` (ready for `uv run inference --vllm.model <dir>` or any HF loader):

```bash
# Export to <ckpt_dir>/weights (streams layers on CPU; add --workers 4 for speed)
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_10 --workers 4

# Or export directly to blockwise FP8 (<ckpt_dir>/weights-FP8) without an intermediate BF16 copy
uv run python tools/convert_dcp_to_fp8.py outputs/my-run/checkpoints/step_10 --workers 4

# Other conversion helpers:
uv run python tools/convert_bf16_to_fp8.py <bf16_dir>   # BF16 HF dir -> blockwise FP8
uv run python tools/convert_fp8_to_bf16.py <fp8_dir>    # FP8 release (e.g. GLM-5-FP8) -> BF16 for training
```

Exports stage into a sibling temp directory and atomically rename into place (LoRA checkpoints are not supported—their broadcast dirs already hold standard `adapter_model.safetensors`). Pass `--overwrite` to replace a custom non-empty target path.

---

## Controlling a Live Run

Enable the HTTP control APIs with `--trainer.control` (port `8040`) and `--orchestrator.control` (port `8041`; SFT uses `--control` on `8040`). Set `PRIME_RL_ADMIN_TOKEN` before launching and pass it in `X-Prime-Admin-Token`:

```bash
export PRIME_RL_ADMIN_TOKEN=$(openssl rand -hex 32)
uv run rl @ rl.toml --ckpt --trainer.control --orchestrator.control --run.name my-run
AUTH="X-Prime-Admin-Token: $PRIME_RL_ADMIN_TOKEN"
```

### Pause or Trigger an On-Demand Checkpoint

Because the trainer and orchestrator step asynchronously, always ask the **trainer first** to get the exact step `N`, then ask the **orchestrator** at that same `step`:

```bash
# 1. Ask trainer to pause (or use {"type": "checkpoint"} to save without stopping)
curl -X POST localhost:8040/v1/actions -H "$AUTH" -H 'content-type: application/json' -d '{"type": "pause"}'
# 2. Poll until state == "accepted" to read step N
curl localhost:8040/v1/actions/<id> -H "$AUTH"
# 3. Tell orchestrator to pause/checkpoint at that exact step N
curl -X POST localhost:8041/v1/actions -H "$AUTH" -H 'content-type: application/json' -d '{"type": "pause", "step": 12}'

# List completed/in-progress checkpoints on disk
curl localhost:8040/v1/checkpoints -H "$AUTH"
```

For `checkpoint`, send the orchestrator request before the trainer finishes step `N` and asks for `N + 1` (otherwise it returns `409`).

### Updating Config Without Restarting

Fetch the live orchestrator config, edit allowed fields, and POST it back—updates take effect after the next shipped step (in-flight rollouts finish with their old settings, and checkpoints preserve live updates on `--resume`):

```bash
curl localhost:8041/v1/config -H "$AUTH" > current.json
jq '{type: "update_config", config: (.config | .max_staleness = 4 | .train.source[0].sampling.temperature = 0.8)}' current.json > update.json
curl -X POST localhost:8041/v1/actions -H "$AUTH" -H 'content-type: application/json' -d @update.json
```

**Fields you can update live** (all others return `422` with the frozen field paths):
- `batch_size`, `max_staleness`
- `concurrency.min_inflight`, `concurrency.max_inflight`
- Per-source `train.source[*].ratio`, `train.source[*].group_size` (must still divide `batch_size`), `train.source[*].sampling` (`temperature`, `max_completion_tokens`, `extra_body` only—`top_p`/`top_k`/`logprobs` are frozen), `train.source[*].algo.length_penalty`
- Per-source `eval.source[*].interval`
- `speculative_decoding.draft` (swaps draft weights at the next policy sync; see [Inference](inference.md#speculative-decoding))

*(Note: edit the per-source entries, not the group-level defaults under `train.*`—group defaults are frozen after startup.)*

---

## Observability: Logs, Dashboard & Monitors

### Run Directory Layout

Every launch (including resumes) creates `<run_dir>/configs/attempt_<n>/` and `<run_dir>/logs/attempt_<n>/`, with `latest` symlinks pointing to the current attempt:

```text
<run_dir>/logs/latest/
├── trainer.log                         # rank 0 (symlink -> trainer/node_0.log on multi-node)
├── orchestrator.log
├── eval.log                            # standalone eval or SFT online eval
├── inference.log                       # symlink -> inference/node_0.log on multi-node
├── trainer/
│   ├── node_*.log                      # multi-node only
│   └── torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log
├── inference/
│   ├── node_*.log                      # multi-node only
│   └── router.log                      # multi-node only
└── envs/{train,eval}/<env_name>.log    # check here first for environment/tool errors
```

Tail everything from the head node:

```bash
tail -F <run_dir>/logs/latest/{trainer,orchestrator,eval,inference}.log
```

### Local Web Dashboard

```bash
uv run dashboard outputs/   # serves at http://localhost:7788 (auto-bumps port if taken)
```

Read-only and safe to run against live training. Pass multiple output directories to compare experiments side by side. Includes six tabs per run:
1. **Overview** — Curated train/eval/stability/performance charts from `metrics.jsonl`.
2. **Metrics** — Every logged metric with regex filtering.
3. **Configs** — Resolved configs and live update revisions.
4. **Traces** — Live and finished rollouts with per-token overlays (advantage, entropy, mismatch KL, IPO mask, loss/content masks), wall-clock timeline, terminal replay, and semantic call graph.
5. **Logs** — Merged component logs.
6. **Reports** — Markdown reports from `<run_dir>/reports/` with verified trace citations.

*(Coding agents can also drive the dashboard via `POST /api/view`; see the `dashboard` skill.)*

### Monitors

- **File monitor (on by default):** Writes `metrics.jsonl` and traces under `<run_dir>/monitors/file/`. Turn off with `--no-monitors.file`. On huge runs where traces dominate disk I/O, sample train groups with `--monitors.file.trace-sample 0.1` (metrics and evals are still saved in full) or disable per-turn live trace files with `--no-monitors.file.live`.
- **Weights & Biases (opt-in):** Enable with `--monitors.wandb` (plus `--monitors.wandb.project` / `--monitors.wandb.name`). Trainer and orchestrator log to one shared W&B run (requires `wandb >= 0.19.9`, incompatible with `offline = true`). Because `prime-rl` logs thousands of per-env/per-stat series, it auto-creates a clean `overview` saved view per project (`overview-v2`, etc., when env sets change).
- **Prime Lab (opt-in, internal):** Enable with `--monitors.prime` (or `[monitors.prime] name = "..."`). Requires `PRIME_API_KEY` (`prime login`) and an allowlisted team (`PRIME_TEAM_ID` / `prime switch`). Set `PRIME_RUNS_MODE=disabled` to suppress platform registration without editing configs, or `RUN_ID=<id>` to attach to a pre-created external run.

---

## Rules of Thumb

- **Smoke test first.** Run `examples/basic/reverse-text/rl.toml` on 2 GPUs before touching a big model or cluster.
- **Keep `batch_size >= 64`.** Below 64, gradients are noisy and per-step overhead dominates (128–512 is good for ablations; 1,024+ for production RL).
- **Keep `group_size >= 8`.** If every rollout in a group passes or fails, advantage collapses to 0 and that task teaches nothing. 16–32 is typical.
- **Name runs you care about.** Pass `--run.name <name>` so you can resume cleanly (reusing a name requires `--resume` or `--clean`).
- **Always `--dry-run` before SLURM.** Catching a bad flag locally takes 2 seconds; catching it in a SLURM queue takes hours.
