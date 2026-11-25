# Algorithms

How `prime-rl` turns rollouts into gradient updates—covering async pipeline staleness, built-in RL and distillation algorithms, losses (IPO and IcePop), curricula, and multi-turn trajectory merging.

## Mental Model: How Algorithms Plug Into the Stack

In `prime-rl`, the **trainer is algorithm-blind**. Every step, the trainer just minimizes a weighted sum of three token-level loss components—**RL** (`rl`), **cross-entropy** (`ce`), and **reference reverse-KL** (`ref_kl`)—each normalized by its own global token count:

$$
\mathcal{L} = \frac{\sum \mathcal{L}_{rl}}{N_{rl}} + \frac{\sum \mathcal{L}_{ce}}{N_{ce}} + \frac{\sum \mathcal{L}_{ref\_kl}}{N_{ref\_kl}}
$$

Why this design is nice:
- Different components never dilute each other's effective learning rate (e.g., adding `echo` observation tokens or mixing an `sft` env beside a `grpo` env in the same batch doesn't weaken the RL gradient).
- Every algorithm lives cleanly on the orchestrator side under `[orchestrator.train.algo]` and just decides two things:
  1. **Who generates rollouts:** the live policy (every algorithm except `sft`), or a frozen teacher (`sft`).
  2. **Per-token signals:** which loss component(s) each token feeds, its advantage (if `rl`), and its reference logprobs (if `ref_kl`).

---

## Choosing an Algorithm (`[orchestrator.train.algo]`)

Pick an algorithm by setting `type`—its class defaults are already the recommended settings:

```toml
[orchestrator.train.algo]
type = "grpo"  # default
```

| `type` | Who samples | Loss used | When to use it |
|---|---|---|---|
| `grpo` *(default)* | Live policy | `rl` | Standard group-relative RL (DR-GRPO: reward minus group mean, no std division). |
| `max_rl` | Live policy | `rl` | [MaxRL](https://arxiv.org/abs/2602.02710): normalizes centered reward by group **mean** instead of std, upweighting hard tasks (`~1/p`). |
| `rae` | Live policy | `rl` | [SPIRAL](https://arxiv.org/abs/2506.24119) Role-Conditioned Advantage Estimation: reward minus a per-agent EMA baseline. Use for **multi-agent self-play** (e.g. `kuhn-poker`). |
| `hierarchical_grpo` | Live policy | `rl` | **Proposer–solver** envs: compares solvers only within the same proposed problem, and proposers against other proposals in the group. |
| `echo` | Live policy | `rl` + weighted `ce` | GRPO on assistant tokens *plus* cross-entropy on environment observation tokens (defaults to `tool` messages at `alpha = 0.1`). |
| `opd` | Live policy | `ref_kl` | [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): policy generates rollouts; trainer minimizes per-token reverse KL to a frozen vLLM `teacher`. |
| `opsd` | Live policy | `ref_kl` | [Self-distillation (SDFT)](https://arxiv.org/abs/2601.19897): live policy scores its own rollouts conditioned on an expert demonstration (no separate teacher server). |
| `sft` | Frozen `teacher` | `ce` | Online hard distillation: a frozen model generates rollouts and the policy trains via CE on its tokens. |

### Mixing Algorithms Across Environments

Each `[[orchestrator.train.source]]` inherits `[orchestrator.train.algo]`, or can override individual fields or pick an entirely different `algo.type`. For example, run `grpo` on math and `echo` on a terminal env in the same run:

```toml
[orchestrator.train.algo]
type = "grpo"

[[orchestrator.train.source]]
name = "math"       # inherits grpo
env.taskset.id = "math"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.train.source]]
name = "terminal"
env.taskset.id = "terminal"
env.agent.harness.id = "bash"
env.agent.runtime.type = "subprocess"
algo.type = "echo"  # this env runs echo instead
```

---

## Algorithm Details & Recipes

### GRPO Options (Length Penalty, Length-Weighted Baseline, Prompt-Mean Loss)

All `grpo`-family algorithms compute `advantage = reward - group_mean` per prompt group (zero-signal rollouts where all RL advantages are 0 are filtered out; by default `orchestrator.constant_trainer_batch_size = true` collects replacements so every trainer batch stays full). Three optional levers help control response length:

1. **Length penalty (`length_penalty`):** Subtracts a pass-rate-scaled penalty from rewards before computing the group baseline, combining group-normalized output tokens (`num_output_tokens_weight`), input tokens (`num_input_tokens_weight`, default `0.1`), and turns (`num_turns_weight`, default `0.1`):
   ```toml
   [orchestrator.train.algo.length_penalty]
   type = "linear"
   ```
   For multi-agent/sub-agent harnesses, use `type = "linear-subagent"` to downweight or exclude sub-agent tokens (`subagent_token_weight = 0.0` by default; `1.0` counts them fully; tracks `subagent_call`, `continuation`, `compaction_attempt`, and `compaction` edges).
2. **Length-weighted baseline (`length_weighted_baseline = true`):** Replaces the plain group mean with a token-count-weighted mean $b = \sum_i L_i s_i / \sum_i L_i$, making per-token advantages zero-mean across the group's tokens instead of across rollouts.
3. **Prompt-mean loss aggregation (`loss_aggregation = "prompt"` on `grpo` / `echo`, default `"token"`):** Weights every prompt group equally ($\frac{1}{T_q}\sum_{t \in q} \mathcal{L}_{rl,t}$) so prompts with huge trajectories don't dominate the gradient (MiMo-V2.6 Eq. 1). Every RL env in a run must use the same `loss_aggregation`.

### ECHO (Training on Observation Tokens)

`echo` adds weighted cross-entropy on non-assistant message roles (needs renderer role attribution). Setting `[orchestrator.train.algo.roles.*]` replaces the default (`tool` at `alpha = 0.1`):

```toml
[orchestrator.train.algo]
type = "echo"

[orchestrator.train.algo.roles.tool]
alpha = 0.25

[orchestrator.train.algo.roles.user]
alpha = 0.05
```

Optionally pass a Python token filter (`[orchestrator.train.algo.filter]`) to drop noisy observation lines (like warnings) per rollout:

```toml
[orchestrator.train.algo.filter]
import_path = "my_module.drop_warnings"

[orchestrator.train.algo.filter.kwargs]
patterns = ["WARNING"]
```

```python
# Returns one boolean keep-mask per trainable branch (False = skip echo CE)
def drop_warnings(rollout, *, patterns: list[str]) -> list[list[bool]]: ...
```

### Hierarchical GRPO (Proposer–Solver)

In a proposer–solver env, one source task spawns multiple proposed problems, and each proposed problem gets multiple solver attempts. Comparing solver attempts across *different* problems mixes wildly different difficulties, and comparing proposers with solvers mixes different jobs.

`hierarchical_grpo` compares **solvers only against sibling attempts on the same proposed problem**, and **proposers against the other proposals in the group**:

```toml
[orchestrator.train.algo]
type = "hierarchical_grpo"
episode_agents = ["solver"]   # roles normalized within each proposed problem

[[orchestrator.train.source]]
name = "proposer-solver"
group_size = 4                # proposed problems per source task
env.n = 4                     # solver attempts per proposed problem
env.taskset.id = "proposer-solver"
env.proposer.harness.id = "null"
env.proposer.runtime.type = "subprocess"
env.solver.harness.id = "null"
env.solver.runtime.type = "subprocess"
```

*(Single-trace comparisons get advantage 0. Use the env's `train_proposer` / `train_solver` flags to freeze one role.)*

### RAE (Multi-Agent Self-Play)

In zero-sum self-play games, players' rewards sum to ~0, so subtracting a group mean washes out signal and turns first-mover asymmetry into constant bias. `rae` tracks a separate exponential moving average baseline per `(env, agent)` (computed *before* folding the current trace in at `decay = 0.95`, so it stays unbiased). Works even with `group_size = 1`:

```toml
[orchestrator.train.algo]
type = "rae"
decay = 0.95

[[orchestrator.train.source]]
name = "kuhn-poker"
env.taskset.id = "kuhn-poker"
env.player0.harness.id = "null"
env.player0.runtime.type = "subprocess"
env.player1.harness.id = "null"
env.player1.runtime.type = "subprocess"
```

Pin one player to a frozen endpoint (`env.player1.model = ...`) for asymmetric play against a fixed opponent; on a single-agent env, `rae` reduces to REINFORCE with an EMA baseline. Baselines live in orchestrator memory and re-warm over `~1 / (1 - decay)` traces after a restart.

### Distillation (`opd`, `opsd`, `sft`)

`prime-rl` only manages one live policy (`[orchestrator.model]`). Any frozen teacher is an external OpenAI-compatible server declared inline on the algorithm (`name` + `base_url`)—launch it yourself (e.g., `uv run inference --vllm.model <teacher> --server.port 8001`).

- **`opd` (On-Policy Distillation):** Live policy generates rollouts; a frozen vLLM `teacher` (required for `prompt_logprobs`) scores each rollout via prefill so the trainer minimizes per-token reverse KL ($\log \pi_{\text{ref}} - \log \pi$):
  ```toml
  [orchestrator.train.algo]
  type = "opd"

  [orchestrator.train.algo.teacher]
  name = "Qwen/Qwen3-32B"
  base_url = "http://localhost:8001/v1"
  ```
- **`opsd` (Self-Distillation / SDFT):** No teacher server needed. Prepends an expert demonstration (read from `info[demo_key]` or top-level field, formatted with `template` containing `{demonstration}`) as a leading system message and scores the rollout verbatim under the live policy:
  ```toml
  [orchestrator.train.algo]
  type = "opsd"
  demo_key = "demonstration"
  ```
- **`sft` (Online Hard Distillation):** Samples directly from `[orchestrator.train.algo.teacher]` (any OpenAI-compatible endpoint) and trains the policy with CE on its completions.

*(Note: `opd`/`opsd` reference scoring runs before curriculum admission, so rejected groups still spend reference prefill compute.)*

### Writing Your Own Algorithm

New algorithms are Python classes registered in code (not config import paths). Subclass `Algorithm` in `src/prime_rl/orchestrator/algo/`, override `score_episode` (per-rollout work/model calls) and/or `score_group` (group-relative credit), add a typed config to `prime_rl.configs.algorithm`, and register it in `ALGORITHM_CLASSES`:

```python
import torch
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.algo.routing import assign_advantages

class MyAlgorithm(Algorithm):
    async def score_group(self, episodes):
        traces = [t for ep in episodes for t in ep.traces]
        rewards = torch.tensor([t.reward for t in traces], dtype=torch.float32)
        advantages = ...  # one scalar per trace (or list per sampled token)
        for trace, adv in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, adv)
```

---

## Async Training & Staleness

By default, trainer and inference overlap by one step: while the trainer updates weights to $\pi_n$, inference is already generating rollouts for step $n+1$ using $\pi_{n-1}$. With balanced GPUs, neither side idles.

![Async pipeline: trainer step n produces $\theta_n$, inference at step n samples with $\theta_{n-1}$](assets/async-pipeline.png)

*(Steps are 1-indexed; policy versions are 0-indexed starting at base model $\pi_0$, so step 1 trains on rollouts from $\pi_0$.)*

Long agentic episodes can span several weight updates before finishing, or sit in a queue waiting to ship:
- **Token staleness** = `(step_version - token_policy_version)`. Engines stamp every token with the policy version that generated it (engines without token labels use dispatch version).
- **Rollout staleness** = max token staleness in the rollout, split in metrics into **in-flight** (updates during generation) and **in-queue** (updates while waiting to ship).
- **`orchestrator.max_staleness` (default `8`):** Tokens older than this are **masked out** of the loss; a rollout is only dropped if *zero* trainable tokens remain. Rollouts are never cancelled for staleness.
- Watch `mismatch_kl_by_staleness/{bin}/mean` (`0`, `1`, `2-3`, `4-7`, …) to tell pure numerical mismatch (bin `0`) apart from staleness-driven drift (higher bins).

---

## RL Losses (`[trainer.loss]`)

`[trainer.loss]` only configures the `rl` component (`ce` and `ref_kl` are fixed). Note: the trainer scales gradients so activation gradients stay independent of batch size and divides parameter gradients by global token counts after backward.

### IPO *(Default)*

**INTELLECT Policy Optimization** is an importance-weighted policy gradient that masks out any token whose **absolute probability** moved by more than `eps` between inference ($\mu$) and trainer ($\pi$):

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_t
\mathbb{1}\!\left(\left|\pi(y_t)-\mu(y_t)\right| \le \epsilon\right)
\tau_A \hat{A}_t \frac{\pi(y_t)}{\mu(y_t)}
$$

- `eps` (default `0.3`): max allowed $|\pi(y_t) - \mu(y_t)|$ before masking.
- `adv_tau` (default `1.0`): advantage temperature multiplier.

Read [IPO](ipo.md) for the intuition and the GLM-4.5-Air ablation behind `eps = 0.3`.

### IcePop

Drops tokens whose importance **ratio** $\pi(y_t)/\mu(y_t)$ falls outside `[ratio_low, ratio_high]` ([arXiv:2510.18855](https://arxiv.org/abs/2510.18855)):

```toml
[trainer.loss]
type = "icepop"
ratio_low = 0.2
ratio_high = 5.0
```

#### IcePop with Score Centering

Subtracts a variance-reducing baseline over the token distribution ([arXiv:2609.20807](https://arxiv.org/abs/2609.20807)):

```toml
[trainer.loss]
type = "icepop"
score_centering = true
score_centering_topk = 128  # used only for untruncated sampling
```

- Requires positive training temperature (and `enable_return_sampling_mask = true` on standalone inference servers; auto-configured by `uv run rl`).
- With **truncated sampling** (`top_p` / `top_k`), [sampling replay](inference.md#sampling-replay) captures the exact support and computes exact centering over it.
- With **untruncated sampling**, captures the top-`score_centering_topk` head and approximates the remaining tail proportionally (floored at `1e-6`).
- Logs `score_centering/{replayed,head_mass,tail_scale,logit_correction_l1}`.

### Custom Loss

Plug in any per-sequence PyTorch function with `type = "custom"`:

```toml
[trainer.loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"

[trainer.loss.kwargs]
clip_eps = 0.2
```

```python
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * inputs.advantages, clipped * inputs.advantages)[inputs.loss_mask].sum()
    return LossOutputs(
        loss=loss,
        metrics={"clip_frac": (ratio != clipped)[inputs.loss_mask].float().mean()},
    )
```

`LossInputs` provides `trainer_logprobs`, `inference_logprobs`, `ref_logprobs` (optional), `advantages`, `loss_mask`, and `loss_weights` (optional). Returned `metrics` are averaged across sequences and logged automatically.

---

## Curricula

Each training source has a **Curriculum** made of one `TaskSampler` (picks which task to run next) and zero or more named `AdmissionGate`s (decides whether a finished group gets trained on; rejected groups are still logged while replacements are sampled). Both are checkpointed and logged under `curriculum/<env>/`.

Built-in pieces:
- **`StandardSampler` (default):** Cycles through finite tasksets in order.
- **`DifficultyPoolSampler`:** Samples with replacement based on each task's latest mean group reward, placing tasks into named threshold pools with custom weights (`weight = 0` disables a pool; unseen tasks start at `1.0`).
- **`AdvRangeGate` (`type = "advantage_range"`):** Rejects a group if *every* trainable token's advantage lies inside `[reject_min, reject_max]`.

```toml
[orchestrator.train.source.curriculum.sampler]
type = "difficulty_pool"

[orchestrator.train.source.curriculum.sampler.pools.hard]
threshold = 0.25
weight = 0.2

[orchestrator.train.source.curriculum.sampler.pools.normal]
threshold = 0.75
weight = 1.0

[orchestrator.train.source.curriculum.sampler.pools.easy]
threshold = 1.0
weight = 0.2

[orchestrator.train.source.curriculum.gates.low_signal]
type = "advantage_range"
reject_min = -0.05
reject_max = 0.05
```

---

## Multi-Turn Trajectories & Renderers

In a 20-turn tool-use rollout, running the trainer separately on every turn wastes $O(T^2)$ compute. Instead, `prime-rl` merges consecutive turns into a single training sequence whenever the **exact-prefix property** holds—meaning turn $t$'s prompt token IDs start with the *exact* token IDs of turn $t-1$'s prompt + completion.

### Best-Effort Interleaving

If the exact prefix holds across turns, compute stays $O(T)$. If anything breaks it (e.g., context compaction, a sub-agent handoff, or a chat template stripping old `<think>` blocks), the orchestrator cleanly splits into a new training sample rather than training on mismatched tokens:

```text
Turns 1–3: exact prefix holds  ──▶ merged into Sample 1
Turn 4:    prefix breaks       (e.g. context compacted or thinking stripped)
Turns 4–5: exact prefix holds  ──▶ merged into Sample 2
Result: 2 clean training samples instead of 5 (and zero logprob drift)
```

### Renderers

Normal `tokenizer.apply_chat_template()` frequently breaks exact prefixes—either through BPE retokenization drift across message boundaries, or because templates (like Qwen3's) intentionally strip earlier `<think>...</think>` blocks on later user turns.

`prime-rl` solves this with [`renderers`](https://www.primeintellect.ai/blog/renderers), which replace Jinja string templates with token-level Python objects (`render_ids`, `parse_response`, and `bridge_to_next_turn`—which appends the new message tokens directly onto the previous turn's token IDs verbatim).

```toml
[orchestrator.renderer]
name = "auto"   # auto-detects from tokenizer; set explicitly for fine-tunes/step checkpoints
```

Hand-coded renderers ship for `qwen3`, `qwen3-vl`, `qwen3.5`, `qwen3.8`, `glm-5`, `glm-4.5`, `minimax-m2`, `deepseek-v3`, `kimi-k2`, `kimi-k2.5`, `nemotron-3`, and `gpt-oss`. Everything else falls back to `DefaultRenderer` (which safely triggers sample splitting when a bridge can't be guaranteed). For external renderers, use `name = "custom"` with `import_path` (see [Training § Dataset Format](training.md#supported-dataset-formats)).
