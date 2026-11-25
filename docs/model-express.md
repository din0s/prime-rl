# ModelExpress Weight Broadcast

`[weight_broadcast] type = "modelexpress"` syncs full FSDP weights from the trainer to vLLM using [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX). Instead of coordinating a giant NCCL collective across both trainer and inference GPUs, the trainer snapshots its shards and inference GPUs pull what they need over RDMA (NIXL).

## Setup

The Python client (`modelexpress==0.8.0.dev20261022`) is included in the GPU extra. Build the matching MX server and Redis backend once on your shared filesystem:

```bash
bash scripts/install_modelexpress.sh
```

`uv run rl` automatically starts a per-run MX server and Redis instance (on the trainer head node for SLURM jobs; logs go to `logs/modelexpress/`). To reuse an already-running service, set `slurm.launch_modelexpress = false` and pass `weight_broadcast.host` and `weight_broadcast.port`.

## Configuration

```toml
[weight_broadcast]
type = "modelexpress"
staging_buffer_bytes = 4294967296  # 4 GB per buffer; omit to stage the whole shard at once
staging_buffers_count = 2          # overlap receiving the next chunk with installing the current one
```

- **`staging_mode`** (trainer side):
  - `COPY_TO_HOST` *(default)* — snapshots trainer shards into CPU RAM so GPU memory stays free.
  - `COPY_TO_DEVICE` — snapshots into extra GPU memory.
  - `IN_PLACE` — zero copy, but trainer weights must already be in transfer dtype and stay unmodified until inference finishes installing them.
- **`staging_buffer_bytes × staging_buffers_count`** (inference side): caps extra receiver memory per GPU. Leaving `staging_buffer_bytes` unset stages a full shard copy on each GPU, which can OOM when `gpu_memory_utilization` is high.
- **`metadata_port`** (default `5655`): base port for trainer NIXL metadata listeners (each local GPU uses `metadata_port + local_rank`).

## How It Behaves & Limitations

- **Warmup:** The first weight sync builds the transfer plan and is slower; subsequent syncs reuse it.
- **Pause & failure:** Inference only pauses *after* the trainer publishes a new version. There is no rollback—if a sync fails, inference stays paused and you must restart trainer and inference together.
- **Shared filesystem:** Trainer rank 0 and the orchestrator only need a shared broadcast directory for lightweight rendezvous markers (weights never touch disk).
- **Not supported:** Rejected at config validation for LoRA, speculative decoding, Dynamo discovery, and SFT online evals. Disaggregated P/D inference is unsupported, and pipeline parallelism is untested.
