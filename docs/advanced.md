# Advanced

Deep-dive reference for supported model architectures, low-precision FP8/MXFP8 training, kernel fusions, multimodal VLM training, LoRA, and building NIXL/UCX for disaggregated P/D inference.

To add a brand-new model architecture, see [Development](development.md#adding-a-new-model).

## Supported Model Families

The trainer only runs `prime-rl`'s custom model implementations (selected automatically from HuggingFace `config.json`). Unsupported architectures fail loudly at startup.

In addition to dense **Llama**, **Qwen3**, and **Qwen3.5**, the following families are supported:

| Family | Example Model IDs | EP | CP |
|---|---|---|---|
| GLM-5 / GLM-5.2 (`glm_moe_dsa`) | `zai-org/GLM-5`, `GLM-5-FP8`, `GLM-5.2`, `GLM-5.2-FP8` | ✅ | ✅ |
| GLM-5.3 Flash, text-only (`glm5_next`) | `zai-org/GLM-5.3-Flash-BF16` | ✅ | ✅ |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B`, … | ✅ | ✅ |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B`, … | ✅ | ✅ |
| Qwen3.5 VLMs | See [Multimodal Training](#multimodal-training) | MoE only | ✅ |
| Qwen3.8 Flash Next VLM | `Qwen/Qwen3.8-Flash-Next` | ✅ | ✅ |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Muse Glimmer (text-only) | `meta-models/Muse-Glimmer-30B` | — | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, `Nemotron-3-Super-120B-A12B`, … | ✅ | ✅ |
| Nemotron H Omni (text-only) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | ✅ | ✅ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | ✅ | ✅ |
| GPT-OSS | `unsloth/gpt-oss-20b-BF16`, … *(SM90/SM100/SM110 + BF16 ckpt required; no MXFP4)* | ✅ | ✅ |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4-Flash-0731` | ✅ | ✅ |

*(Note: GLM-5.2 IndexShare is detected automatically from `indexer_types` in `config.json` with no extra config.)*

---

## Low-Precision Training

Dense linear layers and routed MoE experts are configured **independently**:

1. **Dense linears (`[trainer.model.quantization]`):**
   - `type = "fp8"` — DeepGEMM blockwise FP8 (SM90 / SM100).
   - `type = "mxfp8"` — torchao MXFP8 (SM100; `recipe = "mxfp8_rceil"` or `"mxfp8_rceil_wgrad_with_hp"`).
   - Both skip `lm_head`, routers, and sensitive gates by default (`ignore_patterns`). Add extra modules with `extra_ignore_patterns` (setting `ignore_patterns` replaces defaults).
2. **Routed MoE experts (`[trainer.model.moe.compute]`):**
   - `type = "bf16"` *(default)* — `backend = "torch"` *(default)* or `"sonicmoe"` (requires `uv sync --extra sonic-moe`; currently supports Qwen3 MoE with `gate_up` fusion and bias-free SwiGLU experts).
   - `type = "deepgemm_fp8"` — DeepGEMM FP8 (SM90 / SM100).
   - `type = "mxfp8"` — MXFP8 (requires `prime-kernels`, torchao, SM100).

```toml
# Example: full MXFP8 on Blackwell (SM100)
[trainer.model.quantization]
type = "mxfp8"
recipe = "mxfp8_rceil"

[trainer.model.moe.compute]
type = "mxfp8"
recipe = "mxfp8_rceil"
# Optional: restrict to a subset of layers (default "all", or "85%", or [0, 1, 2, 3])
# apply_to = "85%"

[trainer.model.moe.dispatch]
type = "torch"
transport = "mxfp8"
```

- **Partial-layer MoE precision (`apply_to`):** Applies to `"all"`, a percentage like `"85%"` (first 85% of layers rounded down, counting non-MoE blocks), or an explicit zero-based index list (`[0, 1, 2, 3]`). Unselected layers stay BF16.
- **RL reminder:** Inference precision is *not* inferred from `apply_to`—match your vLLM quantization config explicitly before comparing trainer vs. rollout logprobs.

### Numerics & Precision Guarantees

`prime-rl` keeps a few sensitive spots in FP32 by default so trainer and inference logprobs stay tightly aligned:
- **Router gates:** BF16 operands producing **FP32 logits**, FP32 routing weights, and FP32 accumulation of expert outputs before a single cast back (applies to local, Torch EP, and DeepEP per-rank sums; Mega MoE uses its own combine kernels).
- **LM head:** BF16 operands producing **FP32 logits** (matches vLLM's `enable_fp32_lm_head = true`), BF16 gradients.
- **Model-specific FP32 modules (`fp32_modules`):** Parameters stored and served in FP32 (like GLM-5.3 Flash's hyper-connections and KDA decay `A_log`/`dt_bias`) bypass FSDP's BF16 cast and stay FP32 end-to-end.

### Runtime Parameter Fusions

Enabled by default (`[trainer.model.fusions] enabled = ["gate_up", "qkv"]`; set `[]` to disable; automatically skipped under LoRA):
- `gate_up` packs MoE `gate_proj` + `up_proj` into one tensor, halving grouped GEMMs.
- `qkv` packs attention `q_proj`, `k_proj`, and `v_proj` into one linear layer.

Fusions are purely runtime—checkpoints and exports still use standard canonical weight names, so you can toggle them freely across resumes. Muon orthogonalizes each logical matrix partition separately so training dynamics stay identical. Set `shard_fused_on_dim1 = true` *(experimental, requires `hidden_size % fsdp_size == 0`)* for zero-copy fused checkpointing/loading.

---

## Multimodal Training

Supported built-in VLM families (requiring a custom PrimeRL VLM class):

| Family | `model_type` |
|---|---|
| Qwen3.5 | `qwen3_5` |
| Qwen3.5-MoE | `qwen3_5_moe` |
| Qwen3.8 Flash Next | `qwen4_exp` |

Enable VLM mode by adding `[model.vlm]` (in the shared `[model]` block for RL; see [`examples/extra/vlm/README.md`](../examples/extra/vlm/README.md)):

```toml
[model]
name = "Qwen/Qwen3.5-4B"

[model.vlm]
# freeze_vision_encoder = true  # default; set false to fine-tune vision weights
```

**Gotchas to know:**
- **Text-only checkpoints of VLM families:** If `config.json` sets `language_model_only: true` (no `model.visual` weights), the trainer skips the vision encoder automatically, but vLLM does not—also set `inference.vllm.language_model_only = true` for RL.
- **Vision encoder + LoRA:** `freeze_vision_encoder = false` cannot be combined with LoRA (LoRA freezes all non-adapter parameters, and default LoRA targets don't match vision modules).
- **Precision & monitoring:** Keep FP32 optimizer/reduction (`optimization_dtype = "float32"`, `reduce_dtype = "float32"`) even when serving vLLM in BF16. Watch `mismatch_kl/all/mean` and `is_masked/mean` closely—image preprocessing can introduce small numerical drift. Note that monitors log prompt text only, not raw images.

---

## LoRA Training

Turn on LoRA for SFT or RL by adding `[model.lora]`:

```toml
[model.lora]
rank = 16
alpha = 32
dropout = 0.0
```

- **Target modules:** Defaults cover common attention, MLP, MoE, and Nemotron projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `experts`, etc.); unrecognized names are ignored. Add architecture-specific layers (like Mamba's `in_proj`/`out_proj`) if needed.
- **Weight broadcast:** NCCL broadcast doesn't support LoRA, so `prime-rl` automatically switches to **filesystem** broadcast (writing standard `adapter_model.safetensors` + `adapter_config.json`). Inference serves the adapter under the model name and the base weights as `<model>-base`.

---

## Disaggregated Prefill/Decode Inference

Splitting prefill and decode onto separate node groups prevents long agentic prompts from stalling token generation. Pick your P:D node ratio from your workload shape:

| Workload | Suggested P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean, multi-turn tools) | **3 : 1** | Long growing contexts → prefill-bound |
| Non-agentic (math, chat) | **1 : 2** | Short prompts, long answers → decode-bound |

Check queue depths on live engines to rebalance—if prefill is queuing while decode sits idle, shift nodes toward prefill (and vice versa):

```bash
curl -s http://<prefill_node>:8100/metrics | grep num_requests_waiting
curl -s http://<decode_node>:8200/metrics | grep num_requests_waiting
```

See [`examples/advanced/glm-5.3/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/glm-5.3/swe.toml) (paired with [`examples/advanced/glm-5.3/infer/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.3/infer)) for a full production P/D RL setup.

### Required Setup: Building NIXL Against UCX 1.19.x

The pip wheel for NIXL bundles a UCX build that segfaults (`signal 11` in `libucs.so`) on prefill→decode KV transfers. You **must** build NIXL from source before running disaggregated P/D:

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
uv pip install --reinstall --no-deps deps/nixl_cu13-*.whl
```

This installs UCX 1.19 into `third_party/ucx/` (the SLURM templates automatically prepend it to `LD_LIBRARY_PATH`). **Re-run both commands after every `uv sync`**, since `uv.lock` pins the upstream wheel.
