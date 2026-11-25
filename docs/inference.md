# Inference

`prime-rl` uses **vLLM** for rollout generation, aiming to stay within one stable release of upstream vLLM so features like KV offloading, router replay, and disaggregated prefill/decode work out of the box.

Regardless of how many GPUs or nodes you use, the layout looks the same to clients: **one global router** listens on `inference.server.port` (default `8000`) and fronts all vLLM engines (listening on `inference.backend_port`, default `8100`, plus rank offset). Admin calls like weight updates bypass the router and hit engine ports directly.

Two config sections control everything:
- `[inference.deployment]` — *where* and *how* engines are placed (single node, multi-node, or disaggregated P/D).
- `[inference.vllm]` — pass-through settings forwarded straight to vLLM. Any flag `vllm serve` accepts works here (and on the CLI as `--inference.vllm.<kebab-flag>`, or `--vllm.<kebab-flag>` for standalone `uv run inference`).

---

## Choosing a Deployment Shape

Pick the simplest shape that fits your model and latency budget:

| Shape (`inference.deployment.type`) | Best for | How it scales |
|---|---|---|
| `single_node` *(default)* | Debugging, small models, single-node runs | 1 node with TP / DP / EP |
| `multi_node` | Large models, throughput-focused or single-turn runs | Independent per-node replicas, or multi-node Wide-EP |
| `disaggregated` | Large MoE models on latency-sensitive multi-turn agentic RL | Dedicated prefill nodes + dedicated decode nodes with RDMA KV transfer |

### 1. Single-Node

```toml
[inference.deployment]
type = "single_node"

[inference.vllm]
model = "Qwen/Qwen3-30B-A3B"
enable_expert_parallel = true
tensor_parallel_size = 2
data_parallel_size = 4
```

**Rule of thumb for TP vs. DP:**
- Higher **DP** → higher total throughput, higher per-request latency.
- Higher **TP** → lower per-request latency, saturates with fewer concurrent requests.
- First make sure your model fits with plenty of room left for **KV cache** (shard with TP/EP, or add [KV cache offload](#kv-cache-offload)).

### 2. Multi-Node (Replicas or Wide-EP)

Set `type = "multi_node"` and `num_nodes`:

- **Multi-replica** (`enable_expert_parallel = false`): each node runs an independent vLLM instance, fronted by one router on node 0.
- **Wide-EP** (`enable_expert_parallel = true`): expert parallelism spans across all `num_nodes`, maximizing KV cache headroom on 200B+ MoE models.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference.vllm]
model = "PrimeIntellect/INTELLECT-3"
enable_expert_parallel = true
tensor_parallel_size = 2
data_parallel_size = 8   # 4 local DP ranks per node × 2 nodes
```

### 3. Disaggregated Prefill / Decode (P/D)

Long multi-turn agentic rollouts stall when heavy prefills block active decodes. Disaggregated P/D puts prefill and decode on separate node groups and streams KV cache from prefill to decode over RDMA (NIXL):

```toml
[inference.deployment]
type = "disaggregated"
prefill_nodes_per_replica = 2
num_prefill_replicas = 2
decode_nodes_per_replica = 2
num_decode_replicas = 1
# One inference island = (2×2) + (2×1) = 6 nodes
```

In a multi-node RL run, you can stamp out multiple copies of that entire 6-node island from the top-level `[deployment]` block (`num_infer_nodes` is inferred automatically):

```toml
[deployment]              # top-level RL deployment (not inference.deployment)
type = "multi_node"
num_train_nodes = 4
num_infer_replicas = 3    # 3 islands × 6 nodes = 18 inference nodes total
```

> **Required setup for P/D:** The pip-wheel NIXL's bundled UCX segfaults on prefill→decode KV transfers. You must build NIXL against UCX 1.19.x from source before running P/D—see [Advanced § Disaggregated Prefill/Decode](advanced.md#disaggregated-prefilldecode-inference).

**P/D notes:**
- `prime-rl` automatically sets `deepep_high_throughput` on prefill and `deepep_low_latency` on decode, uses NIXL for KV transfer, and pins single-GPU engines (`tp = 1`) to their nearest PCIe RDMA NIC (`nixl_nic_affinity = true` by default; avoids cross-socket GPUDirect stalls).
- Pass role-specific env vars with `[inference.deployment.prefill_env_vars]` and `[inference.deployment.decode_env_vars]` (layered over `[inference.env_vars]`).

#### Bidirectional KV Transfer (Multi-Turn Speedup)

Normally, on turn 2 of a conversation, prefill has to recompute the KV cache for turn 1's completion even though decode already holds it. With bidirectional transfer (`vllm-router` only; incompatible with HiSparse), decode keeps finished KV blocks briefly and hands prefill a one-time handle so prefill pulls them over NIXL and only computes the *new* tokens:

```toml
[inference.deployment.bidirectional_kv_transfer]
decode_blocks_ttl_s = 480   # how long decode retains finished turn blocks
# router_ttl_s = 450        # defaults to decode_blocks_ttl_s - 30s
```

Each handle is consumed once and dropped by the router at turn start so retries safely fall back to normal prefill.

---

## Router

Select the global router under `[inference.router]` (or `[router]` for standalone `uv run inference`):

- **`vllm-router` (default):** Prime Intellect's [router fork](https://github.com/PrimeIntellect-ai/router). Supports single-node, multi-node, P/D, and router replay. Policies:
  - `sticky_least_loaded` *(default)* — sends a new rollout (`X-Session-ID`) to the least-loaded engine and sticks all later turns there for prefix-cache reuse; releases sessions on completion/error/cancellation.
  - `consistent_hash` — hashes `X-Session-ID` for pure cache affinity (ignores current load).
  - `round_robin` — spreads requests evenly when concurrency is too low for hashing to balance out.
- **`llm-d`:** Upstream [llm-d](https://llm-d.ai) EPP + Envoy proxy (multi-node / P/D SLURM only; does **not** support router replay). Combines prefix-cache affinity with real-time in-flight load balancing (`active-request-scorer`):

```toml
[inference.router]
type = "llm-d"
non_cached_tokens = 16  # P/D: skip remote prefill below this many uncached tokens

[inference.router.scorers]
"prefix-cache-scorer" = 3.0
"active-request-scorer" = 2.0

[inference.router.prefill_scorer_overrides]
"queue-scorer" = 2.0
"kv-cache-utilization-scorer" = 2.0
```

---

## Admin Routes & Auth

`uv run inference` adds orchestrator-only routes on the engine port (`/pause`, `/resume`, `/update_weights`, `/update_weights_from_modelexpress`, `/update_draft_weights`, `/weight_fingerprint/{target}`, `/load_lora_adapter`, `/init_broadcaster`). Because they sit outside vLLM's `--api-key` routes, they use a separate `X-Prime-Admin-Token` header (shared with the trainer/orchestrator/eval control APIs). `/liveness` and vLLM health probes stay open.

- **Managed runs (`uv run rl`, `uv run sft`):** Handled automatically. The launcher generates a token per run (or reuses `PRIME_RL_ADMIN_TOKEN` if you set it), passes the token to the orchestrator/online-eval, and gives engines only its SHA-256 digest (`PRIME_RL_ADMIN_TOKEN_SHA256`) so engines can verify calls without being able to make them.
- **Standalone processes (`uv run inference` + separate `uv run orchestrator`):** Export the same `PRIME_RL_ADMIN_TOKEN` on both sides. Without a token, a standalone server accepts all admin calls with a warning—set `[inference.server] require_admin_token = true` to fail closed instead.
- Customize the env var name via `admin_token_var` (the secret itself never goes into config files or SLURM scripts).

---

## Adaptive Concurrency

Picking a static concurrency cap for RL or eval is almost always wrong: too low leaves GPUs idle, too high causes KV cache thrashing (evicting prefixes only to re-prefill them) and bloats RL staleness—and the sweet spot shifts as rollouts grow longer during training.

`prime-rl` automatically tunes the number of in-flight episodes between `min_inflight` and `max_inflight` (defaults to starting from a safe KV-based estimate; set `initial_inflight` if you know a good starting point, or `max_inflight` if external sandboxes are your scarce resource).

### The Mental Model

Instead of guessing episode cost, the controller measures two live signals from the vLLM engines:

1. **Marginal throughput (`concurrency/elasticity`):** Did adding episodes actually increase tokens/sec? A step up is only kept if throughput grew by at least **30%** of the relative increase in episodes.
2. **Engine queue share (`concurrency/queued_share`):** What fraction of in-flight episodes are sitting parked in engine queues (waiting for capacity or KV loads)?

From those two signals, four simple rules govern the cap:

- **Ramp up & explore:** Starts climbing aggressively at launch (accounting for the fact that young multi-turn episodes haven't reached their full context length yet), then steps up/down around the knee of the throughput curve.
- **Back off on queues:** If `queued_share` stays above 15% (and adding episodes isn't buying throughput anymore), it lowers the cap and lets episodes drain naturally. Above 50% after draining, it cancels the youngest excess episodes.
- **Collapse protection:** Two consecutive windows with a ≥25% throughput drop below the best measurement at that concurrency (e.g., KV thrash or a dying API) trim the cap; a third window with doubled latency or >50% queue cancels excess episodes.
- **Don't outrun the trainer (RL only):** Generating rollouts faster than the trainer can consume them just creates stale data. The dispatcher and controller cap train admission to ~10% above the trainer's measured consumption rate (`trainer_capacity/rollouts_per_s`), while keeping one full batch ready so the trainer never waits. Eval episodes are never throttled by trainer speed.

While engines pause for a weight update, the controller freezes and ignores transient queue spikes for 30 seconds after resuming.

<details>
<summary><strong>Under the hood: exact controller thresholds & maturity estimation</strong></summary>

- **Turnover clock:** Each finished episode advances pool turnover by `1 / inflight`. Startup climb grows up to 3× per turnover (tapering toward 10% queued share), checking windows of at least 3 min and 0.5 turnover.
- **Context maturity:** For long multi-turn episodes, a Kaplan-Meier estimator projects steady-state context (`concurrency/steady_context`) and current pool maturity (`concurrency/maturity`), pacing call-driven growth (up to 1.25× per call turnover, max 3× per 6 min) while keeping maturity ≥80%.
- **Step search:** After climb ends, measurement windows run ≥3 min and 0.5 turnover (1.0 turnover after a step up, with new episodes admitted gradually over 0.5 turnover to prevent waves). Step size scales between 1.15× and 1.5× depending on whether recent steps paid off.
- **Paying override:** While `concurrency/paying` is true (last step still bought throughput), soft queue trims hold off—only the hard cancellation tiers override it.

</details>

---

## Performance & Accuracy Features

### KV Cache Offload

Extend KV cache beyond GPU VRAM into CPU RAM (and optionally local SSD) using `[inference.kv_cache_offload]`. A `cpu` tier is required; `disk` is optional behind it (disk-only is unsupported). Enabling offload automatically enables prefix caching.

- **`type = "native"`:** vLLM's built-in CPU/disk tiering. Self-contained (no extra services). Best for single-node runs. `cpu.num_bytes` is the total CPU pool for the instance.
- **`type = "mooncake"`:** [Mooncake](https://github.com/kvcache-ai/Mooncake) shared distributed KV pool over RDMA (SLURM only). Every inference node contributes `cpu.num_bytes` of DRAM (and optional disk) into one shared pool—so a prefix cached on node A can be loaded over RDMA by node B. Set `cpu.num_bytes = 0` on a node to connect as a reader without contributing RAM.

```toml
# Single-node: native CPU + optional disk
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000  # 128 GB
[inference.kv_cache_offload.disk]
path = "/scratch/kv"

# Multi-node P/D with Mooncake: store KV on prefill nodes, connect decode as reader-only
[inference.deployment.prefill_kv_cache_offload]
type = "mooncake"
[inference.deployment.prefill_kv_cache_offload.cpu]
num_bytes = 1_000_000_000_000

[inference.deployment.decode_kv_cache_offload]
type = "mooncake"
[inference.deployment.decode_kv_cache_offload.cpu]
num_bytes = 0
```

Set `prefill_kv_cache_offload = "None"` or `decode_kv_cache_offload = "None"` to disable offload on one P/D role. Because Mooncake keys blocks by `(model, parallel_rank, content_hash)`, all roles sharing a Mooncake pool must use the exact same TP size, `kv_cache_dtype`, and block size.

### HiSparse *(Experimental)*

For sparse-MLA models with `index_topk` (such as GLM-5.x), HiSparse stores the full KV cache in pinned host RAM and keeps only top-k per-request buffers on GPU, allowing much higher decode concurrency:

```toml
[inference.hisparse]
host_pool_gib = 160   # pinned host RAM per GPU worker (budget: GPUs/node × host_pool_gib)
```

- Under P/D, HiSparse applies to **decode only** (prefill can still use KV offload, but set `inference.deployment.decode_kv_cache_offload = "None"`).
- Without P/D, Mooncake offload works alongside HiSparse, while `native` offload is rejected.
- Explicitly size decode `max_num_seqs` (GPU buffers scale with it). Incompatible with pipeline parallelism, decode context parallelism, and `cudagraph_mode = "FULL"` (P/D decode automatically uses `FULL_DECODE_ONLY`).
- *Status:* Validated statistically on a small random DSA model with P/D + router/sampling replay; not yet A/B validated on 40k+ GLM-5.x quality or preemption/failed KV loads.

### Speculative Decoding

Configure a draft model directly in `[inference.vllm.speculative_config]` (see `examples/specdecode/deepseek-v4/dspark.toml` for a tuned DSpark setup):

```toml
[inference.vllm.speculative_config]
method = "dspark"
model = "/path/to/draft"
num_speculative_tokens = 3
kv_cache_dtype = "fp8"
```

- **Live draft swaps:** In RL, you can update `orchestrator.speculative_decoding.draft` through the control API's `update_config`—engines load the new draft weights at the next policy sync (same architecture/shapes required; failures automatically roll back and increment `speculator/draft_load_failures`).
- **Metrics & routes:** Acceptance logs as `inference/{scope}/spec_decode_acceptance_rate` and `spec_decode_acceptance_length` (includes bonus token). Admin routes: `POST /update_draft_weights` and `GET /weight_fingerprint/{model,draft}`.
- **Replay compatibility:** Works with sampling replay (each accepted token captures its own support). Incompatible with untruncated score centering or adaptive verification with sampling replay.

### Router Replay

On MoE models, small numerical differences can cause the trainer and vLLM to route the same token to *different* experts, inflating trainer–inference KL mismatch by an order of magnitude. **Router replay** records vLLM's exact expert choices and feeds them to the trainer:

```toml
[trainer]
enable_router_replay = true   # automatically enables inference.vllm.enable_return_routed_experts
```

- **Cost:** Adds noticeable HTTP payload overhead—scale up your env worker pools (`serve.pool`) to keep throughput up. Not supported with `llm-d`.
- **With KV offload (native/Mooncake):** Because offloaded KV blocks don't carry routing rows, `prime-rl` only loads offloaded blocks for earlier turns whose routing the client already recorded (before `routed_experts_prompt_start`) and recomputes newer shared prefixes so routing is never missing.

### Sampling Replay

When you sample with `top_p < 1` or `top_k`, vLLM renormalizes token probabilities over only the surviving vocabulary subset. If the trainer computes logprobs over the full vocabulary instead, importance ratios are systematically biased and RL can collapse (DeepSeek V3.2's "Keep Sampling Mask").

**You don't need to set any replay flags.** Just configure truncated sampling on your train source and `prime-rl` enables vLLM's `--return-sampling-mask` and trainer-side renormalization automatically:

```toml
[orchestrator.train.sampling]
top_p = 0.95
# top_k = 128  # optional; defaults to 10000 when only top_p is set
```

**Important details:**
- Training requests require positive temperature and opt in per request (`prl_replay = true`); eval requests on the same server skip mask capture and can use any sampling (including greedy).
- Incompatible with `opd` / `opsd` and with truncation keys smuggled via `extra_body` (validated at startup; frozen-source envs are exempt).
- For standalone `uv run inference`, set `inference.enable_return_sampling_mask = true` (also needed for [score centering](algorithms.md#icepop-with-score-centering)).
- Captured masks live in a GPU buffer of `inference.sampling_support_capacity` entries (default 16M / 128 MB); overflowing raises an explicit error rather than silently truncating.
