# Spike tokens: what does discretizing the population state cost on MC_Maze?

Large language models work over a discrete vocabulary. Neural foundation models mostly don't: NDT3 ([Ye et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/a00000e6a2208172700510bcd69d48e9-Paper-Conference.pdf)) uses categorical cross-entropy over per-channel *counts*, and the VQ-tokenizer work we found ([Tanveer et al. 2026](https://arxiv.org/abs/2609.23907), organoid MEA; [CalM](https://arxiv.org/html/2604.04958), calcium imaging) isn't scored on NLB. This extension asks the narrow question on the same MC_Maze co-smoothing task as the [README](README.md):

**If every 5 ms population state is forced through one of K prototypes, and a causal transformer predicts the next prototype as a token, how much co-bps / vel R² is lost against the same transformer predicting continuous rates?**

Short answer: nothing is lost. A pure token language model beats the continuous model by +0.014 co-bps, and a raw-spike → token model by +0.029. But the ablation below shows the gain comes from **the cross-entropy training target, not from the discrete bottleneck**. When the same discrete head is trained on Poisson NLL instead, it lands exactly on the continuous baseline. A plain Poisson HMM, the 1990s discrete-state model, also matches the continuous causal transformer on co-bps.

All numbers are on the MC_Maze **validation** split (5 ms bins) and scored with `nlb_tools.evaluation.evaluate`, like the rest of the repo. Where a row says "3 seeds", it is mean ± std over init and batch order with the val split held fixed.

## Setup

Every model below shares the causal (autoregressive) transformer from [EXTENSION.md](EXTENSION.md): 4 layers, d=128, causal attention, ~0.8M parameters, the same hyperparameters, and the same val split. Only the input, the output head and the loss change.

**Tokenizer (`aekmeans`, HuBERT-style units).** First, a causal transformer autoencoder is trained on heldin → heldin+heldout rates with Poisson NLL and coordinated dropout ([Keshtkaran & Pandarinath 2019](https://arxiv.org/abs/1908.07896)). Without the dropout it learns to copy the current bin, and co-bps ends up ~0. Then k-means (K=256) runs on its 32-d causal latents. Token `z_t` sees heldin ≤ t. Prototype rates `mu[k]` are the shrunk mean heldin+heldout counts of train bins assigned to k.

**Modes.**
- `lookup`: `rates_t = mu[z_t]`. No sequence model at all.
- `out`: raw heldin spikes (shifted, so ≤ t−1) → logits over K. Then `rates_t = softmax(logits_t) @ mu`.
- `inout`: one-hot tokens in, logits over K out. A pure spike-token LM.
- `*_pois`: identical network, head and `mu`, but trained on Poisson NLL of `softmax(logits) @ mu` against the counts instead of cross-entropy on the next token. **This is the ablation.**

The AR baseline is the same backbone with a continuous softplus head, trained on Poisson NLL.

## Results

| Model (causal, K=256 where tokenized) | input sees | co-bps (3 seeds) | vel R² |
|---|---|---:|---:|
| Continuous AR transformer, Poisson | heldin ≤ t−1 | 0.2703 ± 0.0011 | 0.786 |
| `out`, cross-entropy on tokens | heldin ≤ t−1 | **0.2995 ± 0.0030** | **0.847** |
| `out_pois`, same head, Poisson NLL | heldin ≤ t−1 | 0.2699 ± 0.0038 | 0.797 |
| `inout` (pure token LM), cross-entropy | tokens ≤ t−1 | 0.2847 ± 0.0006 | 0.819 |
| `inout_pois`, same head, Poisson NLL | tokens ≤ t−1 | 0.2636 ± 0.0002 | 0.786 |
| Tokenizer's own autoencoder, unquantized | heldin ≤ t | 0.2872 ± 0.0008 | 0.837 |
| `lookup` (prototype of the current token) | heldin ≤ t | 0.2292 ± 0.0054 | 0.808 |

Discrete-state baseline, a **Poisson HMM** (`run_hmm_mc_maze.py`: Baum-Welch, k-means init, checkpoint on val heldout NLL, heldout emissions marginalized at inference). Single run:

| Poisson HMM | posterior | K=64 | K=256 | K=1024 |
|---|---|---:|---:|---:|
| one-step prediction, causal (same information as AR) | p(z_t \| heldin ≤ t−1) | 0.2598 / 0.710 | **0.2718 / 0.763** | 0.1985 / 0.573 |
| filtered, causal | p(z_t \| heldin ≤ t) | 0.2595 / 0.710 | 0.2666 / 0.760 | 0.1796 / 0.561 |
| smoothed, acausal | p(z_t \| all heldin) | **0.2741 / 0.739** | 0.2677 / 0.801 | 0.1839 / 0.649 |

(co-bps / vel R²)

For reference, from the README and EXTENSION: masked (acausal) transformer v4 is 0.3142 / 0.824 and GRU-v1 is 0.2328 / 0.574. On the MC_Maze 5 ms **test** leaderboard ([EvalAI](https://eval.ai/web/challenges/challenge-page/1256/leaderboard/3188), queried 2026-10-06): SLDS 0.2249 / 0.795, spike smoothing 0.2109, GPFA 0.1872, and STNDT ensemble at the top with 0.3862.

Tokenizer K sweep (`out` / `inout`, single seed): K=64 gives 0.2795 / 0.2703, K=256 gives 0.3006 / 0.2842, and K=1024 gives 0.2632 / 0.2418. K=256 is the sweet spot. At K=1024, `lookup` collapses to 0.092 because the prototypes become too sparse to estimate.

## What it shows

1. **The discrete bottleneck is not where the cost is.** A transformer that only ever sees and emits one of 256 tokens per 5 ms bin (`inout`) beats the same transformer on continuous spikes (0.2847 vs 0.2703; seed std ≤ 0.0011 on both). A raw-spike → token model is the best causal model in this repo on both metrics (0.2995 / 0.847).

2. **The gain comes from the training target.** With identical networks, heads and prototype rates, swapping cross-entropy for Poisson NLL costs 0.030 co-bps on `out` and 0.021 on `inout`. The Poisson-trained discrete head lands on the continuous baseline (0.2699 vs 0.2703). Under Poisson training, the softmax-over-prototypes head is just a constrained rate head, and it does no better than an unconstrained one. NDT3 names CE vs Poisson as untested. This is that test, in one setting.

3. **Why cross-entropy helps is a hypothesis, not a result.** The tokens are targets produced by an autoencoder that was trained to predict heldout neurons. Cross-entropy on those tokens therefore regresses onto a denoised summary of the whole population, where Poisson NLL regresses onto sparse 5 ms counts. Read that way, it is closer to distillation than to "discreteness helps". One observation fits: the `out` student (heldin ≤ t−1) beats its own unquantized teacher (heldin ≤ t, 0.2872) while seeing one bin *less*. Not tested here:
   - a continuous student distilled on the teacher's rates;
   - the extra compute of two-stage training (the tokenizer adds a full training run);
   - whether the tokens' heldout content (the teacher was trained with heldout targets) is what carries the effect.

4. **The 1990s baseline is competitive.** A Poisson HMM with 256 states and one-step-ahead posteriors matches the continuous causal transformer on co-bps (0.2718 vs 0.2703), though with lower vel R² (0.763 vs 0.786). Two things don't behave as expected:
   - The causal filtered posterior, which sees one more bin, scores *lower* than the one-step prediction. Likely cause, not tested: conditioning on 137 neurons makes the posterior over-confident under a model with no within-state variability.
   - K=1024 overfits.

   Our val numbers for smoothing and GPFA are within 0.002 of the leaderboard's test numbers. If that holds here, the HMM's 0.27 on val sits above the SLDS entry (0.2249 on test), but that is a cross-split comparison, not a like-for-like one.

5. **None of this closes the gap to acausal state of the art.** The best causal token model (0.30 val) is below the repo's own masked transformer (0.314 val) and well below the top of the test board (0.386). NDT3 reports 0.278–0.309 causal on MC_Maze at 20 ms bins, which is the same range but not comparable: different bin size, different split, and a larger model (scratch and pretrained variants).

## Limits

- Validation split only. Every comparison here is within one split and one pipeline, so the relative claims (1–4) are what this supports. No absolute leaderboard placement.
- One dataset (MC_Maze), one bin size (5 ms), one small backbone.
- 3 seeds for the transformer rows; the HMM and the K sweep are single runs.
- The tokenizer's autoencoder is part of every token model's training budget, and the AR baseline gets no equivalent. The CE-vs-Poisson rows (2) are matched on this; the token-vs-continuous rows (1) are not.

## Reproduce

After the README setup and `python baselines/prep_tensors_mc_maze.py`. GPU recommended: ~20 min per tokenizer run on an A10G.

```bash
# continuous causal baseline, 3 seeds
for s in 0 1 2; do AR_SEED=$s python baselines/run_ar_transformer_mc_maze.py; done

# token models + CE-vs-Poisson ablation, 3 seeds (also writes the unquantized-autoencoder control)
for s in 0 1 2; do
  VQ_TOKENIZER=aekmeans VQ_K=256 VQ_MODE=lookup,out,inout,out_pois,inout_pois VQ_SEED=$s \
    python baselines/run_vq_ar_transformer_mc_maze.py
done

# Poisson HMM, K = 64 / 256 / 1024
python baselines/run_hmm_mc_maze.py

# score everything in outputs/ (needs ~8 GB RAM for the NWB targets)
python eval_all_mt.py
```

`VQ_TOKENIZER=kmeans` (k-means on causal-EMA spikes) and `vqvae` (end-to-end EMA codebook) are also implemented. Both are worse: best co-bps 0.133 and 0.229 respectively. The tokenizer was the entire gap between "tokens cost half the co-bps" and "tokens win".
