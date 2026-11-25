# IPO (INTELLECT Policy Optimization)

IPO is the default RL loss in `prime-rl`. This page explains the simple idea behind it—masking tokens by **probability difference** instead of **probability ratio**—and the ablation behind the default `eps = 0.3`. For the full loss formula and config knobs, see [Algorithms § IPO](algorithms.md#ipo-default).

## Why Mask by Probability Difference Instead of Ratio?

In async RL, the trainer ($\pi_{\text{train}}$) and the inference server ($\pi_{\text{infer}}$) never agree *perfectly* on every token's probability—due to staleness, MoE routing jitter, and kernel differences. The standard defense keeps importance sampling, but drops tokens where the two disagree too much.

Ratio-based losses like [IcePop](algorithms.md#icepop) mask a token when $\pi_{\text{train}} / \pi_{\text{infer}}$ leaves a band like $[0.2, 5.0]$. The problem is that ratios behave badly at both ends of the probability scale:

- **Rare token (`0.01 → 0.06`):** Ratio is **6×** → IcePop masks it, even though the probability only shifted by `0.05`.
- **Confident token (`0.90 → 0.20`):** Ratio is **0.22** → IcePop *keeps* it, even though the model lost `0.70` of probability mass!

IPO replaces the ratio band with a single absolute probability threshold:

$$
\left|\pi_{\text{train}}(y_t) - \pi_{\text{infer}}(y_t)\right| > \epsilon
$$

```python
probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
is_masked = probs_diff.abs() > eps  # symmetric, independent of advantage sign
```

Intuitively, if you collapse the vocabulary into two events (*this token* vs. *everything else*), $|\pi_{\text{train}} - \pi_{\text{infer}}|$ is just the **total-variation distance** between the two engines. It treats a `0.30` swing in probability mass the same everywhere, with just one knob (`eps`). ([KPop](https://ringtech.notion.site/kpop) independently arrived at the same diagnosis using binary KL.)

## Ablation: IPO vs. IcePop

We compared IPO (`eps = 0.2`, `0.3`) against IcePop (`[0.2, 5.0]`) on a full agentic SWE RL run—everything identical except `[trainer.loss]`:

| Setting | Value |
|---|---|
| Model | `zai-org/GLM-4.5-Air` (100B MoE), BF16 train & inference |
| Task / Eval | `scaleswe` (`bash` harness) / SWE-Bench Verified |
| Cluster | 2 trainer nodes (Muon, CP=4 Ulysses) + 4 single-node TP=8 EP inference replicas |
| Batch & Context | 256 rollouts (16 groups × 16) at 131k context |
| Staleness & Algo | `max_staleness = 64`, no router replay, GRPO + linear length penalty (`0.1`) |

![IPO vs IcePop on GLM-4.5-Air SWE RL: solve rate (top) and trainer–inference KL (bottom)](assets/ipo/ipo-vs-icepop.png)

For the first ~1,000 steps, all runs look identical. Around step 1,200, IcePop's trainer–inference KL spikes >20× and solve rate crashes from ~0.8 to ~0.3—because right before the crash (steps 678–679), IcePop was only masking **0.003%** of tokens (72 out of 2.16M), failing to push back as confident tokens drifted. Both IPO runs stay flat on KL and keep improving.

## Why `eps = 0.3`?

Our first default (`eps = 0.1`) masked ~2% of tokens and caused entropy collapse by step 1,000. Replaying 1.8M sampled tokens shows how the masking rate shifts with `eps`:

![Left: sampled tokens by inference and trainer probability, with the IPO eps = 0.3 band and the IcePop ratio bounds. Right: fraction of tokens masked as a function of eps.](assets/ipo/eps-choice.png)

| Setting | `eps = 0.1` | `eps = 0.2` | **`eps = 0.3`** | `eps = 0.5` | IcePop `[0.2, 5]` |
|---|---|---|---|---|---|
| Tokens masked | 1.98% | 0.64% | **0.25%** | 0.047% | 0.06% |

As the scatter plot shows, IcePop's ratio cone is overly tight near 0 and wide open near 1, whereas IPO's band has uniform width everywhere. Both `0.2` and `0.3` were stable in full runs, and `0.3` kept the highest entropy with the lowest mismatch KL—so `0.3` is the default.

## Reproducing

Configs live in [`examples/advanced/glm-4.5-air/ipo-ablation/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-4.5-air/ipo-ablation) (see the [GLM-4.5-Air README](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-4.5-air) for cluster setup):

```bash
uv run rl @ examples/advanced/glm-4.5-air/ipo-ablation/base.toml @ examples/advanced/glm-4.5-air/ipo-ablation/ipo.toml
uv run rl @ examples/advanced/glm-4.5-air/ipo-ablation/base.toml @ examples/advanced/glm-4.5-air/ipo-ablation/ipo.toml --trainer.loss.eps 0.2
uv run rl @ examples/advanced/glm-4.5-air/ipo-ablation/base.toml @ examples/advanced/glm-4.5-air/ipo-ablation/icepop.toml
```

Compare `mismatch_kl/all/mean` and training solve rate across the three runs.
