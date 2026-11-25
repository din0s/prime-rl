# Eval

`uv run eval` runs one complete evaluation pass over one or more `verifiers` environments and exits. It uses the same worker-pool and concurrency engine as RL training, streams live traces to the local dashboard, and supports resuming interrupted runs.

For periodic evals *during* training, see [Training § Online Evals](training.md#online-evals).

> **For AI coding agents:** see [`skills/eval/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/eval/SKILL.md).

## Quick Start

### Against Prime Inference (Default)

Authenticate with `prime login` (or `PRIME_API_KEY`) and pass a taskset ID directly:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8
```

This evaluates 32 tasks (`-n`), 4 rollouts per task (`-r`), at fixed concurrency 8 (`-c`). The endpoint resolves from `client.base_url` → `PRIME_INFERENCE_URL` → Prime CLI `inference_url` → default Prime Inference URL.

### Against Your Own vLLM Server (or Any OpenAI API)

```bash
uv run inference --vllm.model Qwen/Qwen3-4B
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1
```

### Handy Single-Source CLI Shorthands

When you pass `<taskset-id>` on the command line (without a multi-source TOML):
- `-m <model>` — model name
- `-n N` — `select.limit` (number of tasks)
- `-s` — `select.shuffle`
- `-r K` — `group_size` (rollouts per task)
- `-c N` — pin concurrency (`min_inflight = max_inflight = N`)
- `--env.<field> <value>` — override env settings (e.g., `--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`)

Run `uv run eval -h` to see all shorthands.

## Multi-Source TOML Config

For repeatable or multi-environment evals, put everything in a TOML file (`uv run eval @ eval.toml`). Top-level `[select]`, `[sampling]`, `[env]`, and `group_size` act as defaults that each `[[source]]` inherits and can override:

```toml
model = "Qwen/Qwen3-4B"
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]            # adaptive against local vLLM; pin min == max for external APIs
min_inflight = 8
max_inflight = 256

[sampling]
max_completion_tokens = 2048

[select]                 # shared task selection defaults
limit = 32

[env]                    # shared env defaults
timeout.episode = 7200
retries.max_retries = 3

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
select.include.idx = ["0:30"]
```

**Good to know:**
- **Concurrency:** Set `min_inflight < max_inflight` against a local vLLM server so the [adaptive controller](inference.md#adaptive-concurrency) finds peak throughput automatically. External APIs don't expose vLLM `/metrics`, so pin concurrency with `-c N`. Against a disaggregated P/D deployment, list engine admin URLs in `client.admin_base_url` with matching `"prefill"` / `"decode"` labels in `inference_metrics_roles`.
- **Task selection (`select`):** Applies `include`/`exclude` (by `idx`, `ids`, `keys`, `names`, or newline-delimited files via `include.files.*` / `exclude.files.*`) → `shuffle` → `skip` → `limit`. See `verifiers`' [task selection docs](../deps/verifiers/docs/v1/tasksets.md#selecting-tasks).
- **Worker pools:** Each source spawns its own broker + worker pool on an OS-assigned loopback port (saved to `configs/attempt_N/resolved/envs/eval/<name>.address`), so parallel eval runs on the same machine never collide. Set `serve.address` to connect to an externally managed pool instead.

## Adjusting Concurrency Live

Pass `--control` to expose an HTTP endpoint on port `8040` for shifting concurrency between runs sharing an inference server (if `PRIME_RL_ADMIN_TOKEN` is set, include `X-Prime-Admin-Token`):

```bash
uv run eval @ eval.toml -c 2000 --control

# Check current state
curl localhost:8040/v1/concurrency
# {"target": null, "cap": 2000, "inflight": 2000}

# Lower the cap (in-flight episodes finish normally; new ones wait)
curl -X PUT localhost:8040/v1/concurrency -H 'content-type: application/json' -d '{"target": 1500}'

# Hand control back to the adaptive controller
curl -X PUT localhost:8040/v1/concurrency -H 'content-type: application/json' -d '{"target": null}'
```

## Resuming Interrupted Evals

Give your eval a `--run.name`. If it crashes or gets killed, re-run with `--resume`—every completed `episode.ok` rollout is reused and only failed or missing rollouts re-run:

```bash
uv run eval @ eval.toml --run.name my-eval
# ...interrupted...
uv run eval @ eval.toml --run.name my-eval --resume
```

How resume safety works:
- Previous monitor output is preserved at `monitors/file.attempt_N`; nothing is deleted.
- Tasks match by content hash (`task.hash`), so edited tasks automatically re-run even if their `task.key` didn't change. Duplicate task selections keep separate groups so `pass@k` stays accurate.
- By default, `--resume` checks that the resolved config matches previous attempts exactly (ignoring `resume`) and refuses to mix mismatched runs. Pass `--resume.skip-checks` if you intentionally want to reuse old episodes after changing settings, or use `--clean` without `--resume` to start fresh.

## Dashboard & Monitors

- **File monitor (on by default):** Streams metrics, `plan.json`, finished episodes, and turn-by-turn live traces (`monitors/file/traces/live/<trace_id>.jsonl`) to disk.
- **Local dashboard (`uv run dashboard outputs/`):**
  - **Overview tab:** Shows a live progress grid (landed, errored, and in-flight episodes—click any cell to inspect), headline stats (`avg@k`, `pass@k`, error/truncation rates, turns, latency, cost), per-metric beeswarm/dot plots, and token/phase timing breakdowns (boot, setup, model, harness, finalize, scoring).
  - **Traces tab:** Follows live rollouts turn by turn (with phase badges and pending model calls dimmed) alongside finished episodes. You can also inspect a live trace from the terminal with `uv run python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]`.
  - **Metrics tab:** Charts `eval/<env>/*` summaries plus live `inference/` and `concurrency/` telemetry.
- **W&B (`--monitors.wandb`)** and **Prime Lab (`--monitors.prime`)**: Stream each env's evaluation live. `--monitors.prime` requires `PRIME_API_KEY` (`prime login`), writes its IDs to `monitors/prime/run.json`, and adds a **View on Prime** link to the local dashboard.

## Logged Metrics

Eval metrics live under `eval/<env>/`:

| Metric | Meaning |
|---|---|
| `eval/<env>/all/<agent>/reward/mean` | Mean reward across the epoch |
| `eval/<env>/all/<agent>/is_truncated/mean` | Fraction of rollouts that hit the length limit |
| `eval/<env>/all/<agent>/is_timeout/mean` | Fraction of rollouts stopped by a stage timeout |
| `eval/<env>/all/<agent>/has_error/mean` | Fraction of rollouts that raised an exception |
| `eval/<env>/all/seq_len/mean` | Mean episode length in tokens |

Any custom metrics returned by your environment (e.g., `correct_answer`, `format`) appear under the same prefix.
