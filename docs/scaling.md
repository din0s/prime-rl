# Scaling

How to go from a single GPU to a 1,000-GPU SLURM cluster—covering GPU placement, trainer parallelism (FSDP2, EP, CP), memory vs. speed recipes, weight broadcasts, and synthetic benchmarking.

For inference topologies, see [Inference](inference.md). For general training workflows, see [Training](training.md).

## Single-Node vs. Multi-Node Placement

Every entrypoint (`rl`, `sft`, `inference`) takes a `[deployment]` block:
- **`type = "single_node"` (default):** runs everything locally.
- **`type = "multi_node"`:** submits a SLURM job (see [SLURM](#slurm)) that wires up rendezvous endpoints, ports, and shared paths across nodes.

### Single-Node RL GPU Split

By default, `uv run rl` uses 1 GPU for inference and 1 GPU for training. To split an 8-GPU node 6/2 (inference GPUs are assigned first from `CUDA_VISIBLE_DEVICES`, trainer GPUs next):

```bash
uv run rl @ rl.toml \
  --deployment.num-infer-gpus 6 \
  --deployment.num-train-gpus 2 \
  --inference.vllm.data-parallel-size 6
```

To run two A/B experiments side by side on the same box, pin `CUDA_VISIBLE_DEVICES` and give the second run a different inference port:

```bash
# Terminal 1 (GPUs 0–1, port 8000)
CUDA_VISIBLE_DEVICES=0,1 uv run rl @ rl.toml --run.name exp1

# Terminal 2 (GPUs 2–3, port 8001)
CUDA_VISIBLE_DEVICES=2,3 uv run rl @ rl.toml \
  --inference.server.port 8001 \
  --orchestrator.model.client.base-url http://localhost:8001/v1 \
  --run.name exp2

# Compare both live
uv run dashboard outputs
```

### Single-Node SFT

`uv run sft` launches `torchrun` internally—just set the GPU count under `[deployment]`. Only drop down to raw `torchrun` if you need a custom launcher:

```bash
uv run torchrun --nproc-per-node 8 --local-ranks-filter 0 \
  src/prime_rl/trainer/sft/train.py @ sft.toml
```

---

## Trainer Parallelism & Memory Levers

### FSDP2 (Data & Weight Sharding)

FSDP2 is always on and fully shards parameters, gradients, and optimizer states across ranks by default.

| Knob | What it does |
|---|---|
| `trainer.model.dp_replicate` | Enables Hybrid Sharded Data Parallel (HSDP): replicate weights across `dp_replicate` islands and FSDP-shard inside each island. Useful on huge clusters where pure FSDP communication bottlenecks. |
| `trainer.model.reshard_after_forward` | Default `true` (frees memory after forward, re-gathers in backward). Set `false` for higher speed when VRAM isn't tight. |
| `trainer.model.offload` | `"optimizer"` *(default)* offloads optimizer state to CPU between steps; `"full"` keeps BF16 weights on GPU and runs CPU AdamW or SignSGD chunks during backward (disables grad clipping; SignSGD halves host RAM to 8 bytes/param); `"none"` keeps everything on GPU. |

### Expert Parallelism (EP) for MoE

EP shards MoE experts across GPUs instead of all-gathering every expert on every rank.

`trainer.model.ep` defaults to `"auto"`—at startup it picks the largest degree up to 8 that divides both the FSDP island size (`world_size // dp_replicate`) and `num_experts` while staying a multiple of `cp` (resolves to `1` for dense models). Override with an integer if needed:

```toml
[trainer.model]
ep = 8  # must divide num_experts

[trainer.model.moe.dispatch]
type = "torch"      # "torch" (default), "deepep", or "mega_moe"
transport = "bf16"  # or "mxfp8" on SM100
```

Choose the dispatch backend under `[trainer.model.moe.dispatch]`:
- **`torch` (default):** PyTorch all-to-all. Pair with `[trainer.model.moe.compute]` (`bf16`, `deepgemm_fp8`, or `mxfp8`—see [Advanced](advanced.md#low-precision-training)).
- **`deepep`:** Custom DeepEP kernels (`num_sms` controls SMs dedicated to comm, default `32`).
- **`mega_moe`:** Persistent fused dispatch + SwiGLU MLP + combine kernels from Prime Intellect's closed-source `prime-mega-moe` wheel (`uv pip install --no-deps <wheel>`). Requires `ep > 1`, `nvcc` 13 in `CUDA_HOME`, bias-free gated experts, `hidden_size % 256 == 0`, and `moe_intermediate_size % 128 == 0` (`% 256` for `nvfp4`). Precision is set directly on dispatch (`bf16`, `fp8`, `fp8_bwd` on SM90; `mxfp8`, `mxfp8_qat`, `nvfp4` on SM100—`[trainer.model.moe.compute]` is ignored).

<details>
<summary><strong>Mega MoE tuning: <code>num_reserved_sms</code> and LLEP load balancing</strong></summary>

Because Mega MoE kernels are persistent grids synchronized across ranks, NCCL collectives (FSDP all-gathers/reduce-scatters) must fit into the SMs left free (`num_reserved_sms`, default `16`, or `32` with LLEP) or the two deadlock. `prime-rl` automatically caps `NCCL_MAX_CTAS` to `num_reserved_sms // 2` (`// 3` with HSDP) and rejects higher manual values. Symmetric buffers size from `max_tokens_per_rank` (defaults to a rank's CP micro-batch share).

```toml
[trainer.model.moe.dispatch]
type = "mega_moe"
precision = "bf16"
num_reserved_sms = 16

[trainer.env_vars]
CUDA_HOME = "/usr/local/cuda-13.0"
PATH = "/usr/local/cuda-13.0/bin:${PATH}"
```

**Least-Loaded Expert Parallelism (LLEP):** Add `[trainer.model.moe.dispatch.llep]` (supported on `bf16` SM90 and `mxfp8`/`mxfp8_qat`/`nvfp4` SM100 with an LLEP-enabled wheel). When routing hot-spots one rank, overloaded routes are dynamically shipped to underloaded ranks along with a temporary NVLink copy of those experts' weights—bitwise identical to plain Mega MoE, with much smaller symmetric buffers:

```toml
[trainer.model.moe.dispatch.llep]
alpha = 1.05          # rank considered full at alpha × mean routes
min_chunk = 1024      # minimum routes worth shipping an expert's weights for
lam = 1.1             # minimum busiest/mean imbalance before offloading triggers
max_foreign = 8       # max borrowed experts per rank
max_recv = 1.3        # receive buffer size as multiple of a rank's own routes
chunk_tokens = 16384  # optional: chunk long sequences across kernel calls
```

</details>

### Context Parallelism (CP) for Long Sequences

CP splits a single long sequence across multiple GPUs so activations fit in memory:

```toml
[trainer.model]
attn = "auto"  # FA3 on Hopper, FA4 on datacenter Blackwell, FA2 otherwise
cp = 2         # sequence-parallel degree
```

- **Standard & hybrid models (Ulysses CP):** All-to-all redistributes sequence shards into head shards for local attention (and similarly for linear-attention/Mamba layers in Qwen3.5 and Nemotron-H). `cp` must divide `num_attention_heads` (and be a multiple of `num_key_value_heads` for GQA replication; on Nemotron-H it must also divide `mamba_num_heads` and `n_groups`).
- **Sparse-attention models (GLM-5 DSA, DeepSeek-V4):** Keep queries sequence-sharded without head sharding (GLM-5 all-gathers keys; DeepSeek-V4 exchanges only sliding-window/compressed tokens).

### Activation Checkpointing (AC) & Offloading

Both **full AC** (`trainer.model.ac`) and **AC CPU offloading** (`trainer.model.ac_offloading`, `max_inflight_activations = 5`) are **enabled by default**. AC offloading often cuts peak VRAM by 30–40% for only ~3–5% throughput loss. Disable either by setting it to `"None"`.

- **Selective AC:** Keeps cheap/memory-light recomputes while retaining expensive GEMMs, grouped GEMMs, attention, and collectives (smaller throughput hit than full AC):
  ```toml
  [trainer.model.ac]
  mode = "selective"
  ```
  Override what gets retained via `targets` (op names like `"aten::mm"`, namespaces like `"prime_rl_collectives"`, or module-scoped globs like `"prime_rl::fp8_blockwise_mm@mlp.shared_expert"`—module globs require `trainer.model.compile`).
- **Token-chunked AC *(experimental, Nemotron-H MoE only)*:** Set `trainer.model.ac.token_chunk_size = 8192` to checkpoint MoE blocks one token chunk at a time on long sequences (requires `ep == 1` or `ep == cp`).

### Fused LM Head Chunking

With 100k+ vocabularies, materializing the full `[batch * seq, vocab]` logits tensor burns gigabytes of VRAM. `prime-rl` defaults to streaming through `8192` tokens at a time with a fused linear + logprob/entropy kernel:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = 8192        # lower if VRAM is tight; "disabled" for vanilla head
```

---

## Two Practical Recipes

### Recipe 1: Fitting a Huge MoE on Limited GPUs (Memory-Tight)

Defaults already enable full AC, AC offloading (`5`), `torch.compile`, optimizer CPU offload, fused LM head (`8192`), and auto-EP. To squeeze even harder, tighten AC offloading and add CP:

```toml
[trainer.model]
ep = 8
cp = 2

[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 1
```

### Recipe 2: Maximum Trainer Speed (Enough GPUs to Avoid Offload)

Because trainer defaults optimize for memory, turn off CPU offloading and enable `fullgraph` compilation once you have enough GPUs to hold state on-device:

```toml
[model]
offload = "none"
ac_offloading = "None"

[model.compile]
fullgraph = true
```

Scale out with more nodes rather than CP (`cp = 1`) and run `micro_batch_size = 1` with full-length packed rows. Reference configs: [`examples/advanced/qwen3-235b-a22b/sft-max-perf.toml`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/qwen3-235b-a22b) (~46% MFU on 4×8 B300) and [`examples/advanced/glm-5.3/sft/h200/16k-12-node.toml`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.3/sft/h200).

---

## SLURM

Adding a `[slurm]` table to any `rl`, `sft`, or `inference` config automatically submits to SLURM—there is no separate launcher command.

> **Shared filesystem required:** Keep your `prime-rl` repo and `.venv` on a shared filesystem visible to every node (the generated sbatch script runs `uv sync` once on the batch node). JIT caches (`TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE`) default to node-local `/tmp` so ranks never corrupt a shared cache.

Create a thin overlay with your SLURM and multi-node settings:

```toml
# slurm_overlay.toml
output_dir = "/shared/outputs/my-rl"

[deployment]
type = "multi_node"
num_train_nodes = 2
num_infer_nodes = 1     # optional in RL if inference.deployment sets topology; in SFT, only for online evals
gpus_per_node = 8

[slurm]
job_name = "my-rl-run"
```

Dry-run first to inspect the generated sbatch script under `<run_dir>/launcher/rl.sbatch` (SLURM logs go to `<run_dir>/launcher/logs/`), then submit:

```bash
uv run rl @ base_rl.toml @ slurm_overlay.toml --dry-run
uv run rl @ base_rl.toml @ slurm_overlay.toml
```

Need custom modules or partitions? Copy a Jinja2 template from [`src/prime_rl/templates/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/templates) and pass `--slurm.template-path my_template.sbatch.j2`.

For complete multi-node examples, see [`configs/advanced/nemotron-3-super/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/configs/advanced/nemotron-3-super/swe.toml), [`examples/advanced/glm-5.3/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.3), and [`examples/advanced/glm-4.5-air/swe-2-node.toml`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-4.5-air/swe-2-node.toml).

### Weight Broadcast Transports

By default, weights sync over NCCL (or filesystem for LoRA). At scale, two RDMA-backed transports avoid a cross-cluster NCCL collective between trainer and inference:

- **ModelExpress (`[weight_broadcast] type = "modelexpress"`):** See [ModelExpress](model-express.md).
- **NIXL (`[weight_broadcast] type = "nixl"`):** Receiver-driven RDMA pulls over InfiniBand/CUDA via UCX, with job-scoped ModelExpress/Redis metadata exchange on the trainer head node (or reuse an existing server with `slurm.launch_modelexpress = false` + `weight_broadcast.{host,port}`).
  - One-time shared-FS setup:
    ```bash
    bash scripts/install_nixl_from_source.sh
    uv pip install --reinstall --no-deps deps/nixl_cu13-*.whl
    bash scripts/install_modelexpress.sh
    ```
  - Automatically picks each GPU's nearest active InfiniBand port (unless `UCX_NET_DEVICES` is set) and staggers trainer source ranks across inference workers.
  - Set `weight_broadcast.overlap_transfer_and_replay = true` to double-buffer transfer arenas and overlap receiving the next weight group with replaying the current one (costs one extra largest-group buffer per GPU).

---

## Benchmarking Trainer Throughput

Before committing to a multi-day run, measure throughput, MFU, and peak memory in seconds using synthetic data (`--data.fake`):

```bash
# Benchmark SFT trainer
uv run sft @ sft.toml --data.fake --max-steps 4

# Benchmark RL trainer alone (no inference needed)
uv run torchrun --nproc-per-node=8 src/prime_rl/trainer/rl/train.py @ train.toml --data.fake --max-steps 4
```

For realistic packed RL sequences, add `--data.fake.generate-samples` (and optionally `--data.fake.doc-len N` for fixed-length documents inside each packed row). Read `perf/throughput`, `perf/mfu`, `time/step`, and `perf/peak_memory` from console logs or `<run_dir>/monitors/file/metrics.jsonl`—always ignore step 1 (compile/JIT warmup).
