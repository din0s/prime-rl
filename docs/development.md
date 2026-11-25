# Development

Guide for contributing to `prime-rl`: running tests locally, understanding GPU CI, and bringing up new text or multimodal model architectures without booting a 100B+ cluster.

## Pre-Commit Setup

Install pre-commit hooks before your first commit so `ruff` lints and formats staged Python files automatically:

```bash
uv run pre-commit install
```

## Testing

The test suite has three layers:
- **`tests/unit/`** — Fast, isolated tests (config validation, loss/advantage/packer math, weight conversion roundtrips). GPU-requiring tests carry `@pytest.mark.gpu`.
- **`tests/integration/`** — End-to-end RL and SFT runs on tiny models through inference + orchestrator + trainer.
- **`tests/regression/`** — Nightly invariance tests (`test_parallel_invariance.py` checks DP/CP/EP matches 1-GPU step 1) and evals vs. baselines.

### Running Tests Locally

```bash
uv run pytest -v -m "not gpu"                              # CPU-only (fast; mirrors CPU CI)
uv run pytest tests/unit tests/integration -m tiny -v      # Fast GPU subset (mirrors Tiny GPU CI)
uv run pytest -v -m gpu                                    # Full GPU suite
uv run pytest tests/integration/test_reverse_text.py -vvs  # Single integration test

# Run the exact Modal GPU CI suites directly from your shell
uvx modal run scripts/modal_ci.py --suite tiny
uvx modal run scripts/modal_ci.py --suite full --group rest

# Required once if running dashboard integration smoke tests
uv run playwright install --only-shell chromium
```

### Pytest Markers

Defined in `pyproject.toml` (`--strict-markers`):
- `gpu` — Needs CUDA. Unlisted files run in the full suite's 2-GPU `rest` group in `scripts/modal_ci.py` (give multi-minute files their own group + `gpu-full.yml` matrix entry).
- `tiny` — Fast GPU subset run on every push to a non-draft PR (~2.5 min total on Modal). Unlisted files default to the 1-GPU group—add a dedicated group if a test needs 2 GPUs or >1 min (check with `--durations=0`).
- `slow` — Expensive tests you can skip locally via `-m "not slow"`.

### CI Workflows

- **[`tests.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/tests.yml):** Runs CPU unit tests on every PR/push, verifies `prime-rl-configs` imports cleanly with zero heavy ML deps, and runs a hosted `gsm8k_eval` job when relevant files change.
- **[`gpu-tiny.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu-tiny.yml):** Runs `Tiny (GPU)` on Modal for every push to a non-draft PR (newer pushes cancel older runs; draft PRs skip GPU jobs).
- **[`gpu.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu.yml):** Runs `Tiny (GPU)` followed by the full GPU suite (`gpu-full.yml`) on every merge to `main`, or on any PR when you apply the **`gpu-ci`** label (the workflow removes the label once started; re-apply to re-run). Full runs execute one at a time globally to keep GPUs free for Tiny jobs. Fork PRs cannot run GPU CI directly.
- **[`regression.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/regression.yml):** Daily at 03:00 PST (and manual dispatch) for parallel invariance (4×H100 Modal) and baseline eval checks (posts to Slack).

---

## Adding a New Model

Bringing up a new architecture takes three steps—iterating on a ~0.5B random-weight "mini" clone on 1–2 GPUs first so you catch conversion and pipeline bugs in minutes instead of hours.

### 1. Implement the Model

Create `src/prime_rl/trainer/models/<arch>/`, using `glm4_moe/` or `qwen3_moe/` as a template:

- **Config:** Subclass HF `PretrainedConfig` so saved `config.json` files stay loadable by vLLM.
- **Model classes:** An inner model and a `*ForCausalLM`, both subclassing `PreTrainedModelPrimeRL` (`models/base.py`), returning `models/base.py` output dataclasses, and ending `__init__` with `self.post_init()`.
- **Weight tying:** Set `supports_tied_word_embeddings = True` on `*ForCausalLM` if the architecture supports tying `lm_head` to input embeddings (`config.tie_word_embeddings` controls whether a checkpoint ties).
- **Initialization:** Override `_init_weights` (calling `super()._init_weights(module)` first) for custom parameters, and `init_buffers_post_meta` for buffers not stored in checkpoints.
- **State-dict conversion:** Implement `conversion_chain`, `is_hf_state_dict`, and `is_prime_state_dict` to map cleanly between HF checkpoints and trainer layout.
- **Registration:** Register the `(config, model)` pair in `_CUSTOM_CAUSAL_LM_MODELS` (`models/__init__.py`) and call `AutoConfig.register`.

### 2. Add a Mini Preset

Add a small ~0.5B preset in [`scripts/mini_moe.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/mini_moe.py):

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(hidden_size=1024, num_hidden_layers=24, n_routed_experts=8, ...),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
}
```

### 3. Build, SFT-Warmup & Smoke Test

```bash
# 1. Generate mini model + verify lossless HF <-> prime-rl conversion roundtrip
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
# (Re-check conversion later without regenerating: add --verify-only)

# 2. Quick SFT warmup so the distribution is non-uniform (loss ~12 -> ~2.5)
#    Or skip and use the prebuilt checkpoint: samsja/mini-glm-moe
uv run sft --model.name ./mini-glm-moe --max_steps 200 --optim.lr 1e-4 --ckpt

# 3. Run full RL stack end-to-end
uv run rl @ configs/ci/integration/reverse-text-moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --inference.vllm.gpu-memory-utilization 0.7 \
  --inference.vllm.max-model-len 2048
```

Verify three things: no crashes, finite non-zero KL, and healthy (non-NaN) loss. (Don't expect reward to climb in 20 steps on a random mini model.)

### Merge Requirements for New Models

1. Satisfies the modeling contract above and passes the mini smoke test.
2. PR description includes a **20-step KL mismatch table** on the `math` environment (`batch_size = 64`) where **every entry is `< 0.015`** (proving numerical alignment with vLLM).

---

## Adding a Custom VLM

VLM training (`[model.vlm]`) only supports registered PrimeRL VLM classes. Use `models/qwen3_5/` as the reference implementation and follow this order:

1. **Text model first:** Complete the text `*ForCausalLM`, state-dict conversions, and `< 0.015` KL-mismatch table.
2. **Composite VLM body (`*VLMModel`):** Hold the vision encoder at `model.model.visual` and the language model at `model.model.language_model` (FSDP, freezing, AC, compile, MoE runtime, and weight broadcast expect those exact attribute paths). Embed tokens, run vision, scatter image embeddings into placeholder slots, and compute MRoPE 3D positions from `mm_token_type_ids` when `image_grid_thw` is present (falling back to plain `position_ids` for text-only rows). Dispatch VLM vs. text path inside the unified `*ForCausalLM` based on config type.
3. **Keep FSDP/EP symmetric on text-only micro-batches:** Always run the vision encoder on dummy pixels and add `image_embeds.sum() * 0.0` to `inputs_embeds` so ranks never hang when the encoder is unfrozen.
4. **Honor packed boundaries:** Accept `seq_lens` and `seq_lens_are_pre_shard` in `forward()`, respect per-document boundaries in varlen FlashAttention (`cu_seqlens`) and linear-attention state resets, and set `supports_packed_multimodal_training = True` once verified.
5. **Register:** Add the composite `model_type` to `_CUSTOM_VLM_MAPPING` (`models/__init__.py`) and implement `materialize_images(refs, processor)` (see `vision.py` in Qwen3.5 / Qwen3.8).
6. **Optional CP:** Attach `CPContext` to CP-aware modules and shard embeddings/positions *after* vision scatter (trainers defer sharding to the model for image batches).
7. **Validate:** Text KL-mismatch table + a real multimodal SFT run and RL run (using the `color-codeword` environment).
