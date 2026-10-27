# Prototype distillation: training on a quantized teacher's rates lifts co-smoothing on MC_Maze and MC_RTT

> **Draft, 2026-10-27.** Final pass planned for the week of 2026-11-09.

[TOKENS.md](TOKENS.md) started as a question about spike tokens and ended somewhere else: a plain continuous transformer trained on a tokenizer's **prototype rates** instead of the raw spike counts beat every token model. This page follows that result up. It asks three things:

1. Does it lift the strongest model in the repo, the acausal masked transformer, and not only the small causal one?
2. Is it the *prototype* teacher that matters, or would any soft target (the model's own rates, temporally smoothed counts) do the same?
3. Does it hold on a second dataset and a second monkey?

Short answer: yes, the prototype teacher specifically, and yes. On MC_Maze the masked transformer goes from **0.3139 to 0.3306 co-bps** (+0.017) and vel R² from 0.823 to 0.883. On MC_RTT, with the same configs and no retuning, it goes from **0.1366 to 0.1615** (+0.025). Self-distillation and smoothed counts both *hurt*.

All numbers are on the NLB **validation** split (5 ms bins), scored with `nlb_tools.evaluation.evaluate`, mean ± std over 3 seeds (init, masks and batch order; the val split is held fixed).

## Method

**Teacher (`aekmeans`, as in TOKENS.md).** A causal transformer autoencoder is trained heldin → heldin+heldout rates with Poisson NLL and coordinated dropout ([Keshtkaran & Pandarinath 2019](https://arxiv.org/abs/1908.07896)). k-means (K=256) runs on its 32-d latents, which assigns every 5 ms training bin a token `z_t`. The prototype rates `mu[k]` are the shrunk mean heldin+heldout counts of the training bins assigned to k. The teacher's target for bin t is `mu[z_t]`: one of 256 population rate vectors.

**Student.** The masked transformer v4 from [EXTENSION.md](EXTENSION.md), unchanged: same architecture, masking, hyperparameters and schedule. The only change is the training target. Wherever its loss compared rates to counts (masked heldin entries, all heldout entries), it now compares them to the teacher's rates. Checkpoints are still selected by val Poisson NLL against the **real counts**.

**Targets compared** (all train the same student):
- `counts`: the spike counts (plain MT v4).
- `dist_mu_mt`: prototype rates `mu[z_t]`.
- `dist_ae_mt`: the unquantized autoencoder's rates (same teacher, before k-means).
- `self`: self-distillation. Train MT v4 on counts, then train a fresh MT v4 on that model's rates.
- `smooth50`: counts Gaussian-smoothed in time, σ = 50 ms (acausal).

## Results

**MC_Maze** (monkey Jenkins, delayed center-out reaching through a maze; 137 heldin / 45 heldout):

| Masked transformer v4 trained on | co-bps (3 seeds) | vel R² |
|---|---:|---:|
| counts | 0.3139 ± 0.0018 | 0.823 |
| **prototype rates `mu[z_t]` (`dist_mu_mt`)** | **0.3306 ± 0.0015** | **0.883** |
| autoencoder rates (`dist_ae_mt`) | 0.3063 ± 0.0040 | 0.881 |
| own rates (self-distillation) | 0.2961 ± 0.0032 | 0.853 |
| counts smoothed, σ = 50 ms | 0.2646 ± 0.0094 | 0.726 |
| *teacher: autoencoder, unquantized* | *0.2872 ± 0.0008* | *0.837* |

The same comparison on the small causal transformer (from TOKENS.md): counts 0.2703, `dist_ae` 0.2908, `dist_mu` 0.3117.

**MC_RTT** (monkey Indy, continuous random-target reaching, no trial structure; same configs, no retuning):

| Masked transformer v4 trained on | co-bps (3 seeds) | vel R² |
|---|---:|---:|
| counts | 0.1366 ± 0.0021 | 0.305 |
| **prototype rates `mu[z_t]` (`dist_mu_mt`)** | **0.1615 ± 0.0051** | **0.366** |
| own rates (self-distillation) | 0.1197 ± 0.0018 | 0.294 |
| *teacher: autoencoder, unquantized* | *0.1413 ± 0.0041* | *0.278* |

Per-seed scores: [`results/`](results/) (`eval_all_2026-10-15.txt`, `eval_all_2026-10-16.txt`, `eval_mc_rtt_2026-10-19.txt`).

## What it shows

1. **Prototype distillation lifts the strongest model, not only the weak one.** +0.017 co-bps and +0.06 vel R² on the masked transformer, with a seed spread of ~0.002. On the causal transformer it was +0.041. The gain shrinks as the student gets stronger, but does not vanish.

2. **It is the prototype teacher, not soft targets in general.** Two generic soft targets were tested on the same student, and both lose to counts. Self-distillation costs 0.018 co-bps (though it raises vel R²). Temporal smoothing costs 0.049. So the gain does not come from "any smoother target regularizes". The unquantized autoencoder's rates also lose to counts on the masked transformer (−0.008), while the quantized ones win (+0.017). Quantization is what turns the teacher from a worse target into a better one.

3. **The student beats its teacher.** On both datasets `dist_mu_mt` beats the autoencoder that produced its targets by a wide margin (0.331 vs 0.287 on MC_Maze, 0.162 vs 0.141 on MC_RTT). The student isn't copying a strong model; it is learning from a better-conditioned target than raw 5 ms counts.

4. **It replicates on a second dataset and monkey, with a larger gain.** MC_RTT has no trials, no maze and a different subject and array. With every hyperparameter carried over from MC_Maze, `dist_mu_mt` gains +0.025 co-bps there, against +0.017 on MC_Maze. Self-distillation again hurts.

**Why quantized targets distill better is untested.** One candidate: a prototype is a mean over many training bins in the same population state. That averages out per-bin Poisson noise much more than one model's rate estimate does, while staying sharp across states, which temporal smoothing doesn't.

## Prior work and novelty

Literature check run 2026-10-27, after the checks behind TOKENS.md (2026-09-30, 2026-10-02).

- **Closest prior: BLEND** ([ICLR 2025](https://arxiv.org/abs/2410.13872)). This is teacher-student distillation on the same NLB'21 datasets (MC_Maze, MC_RTT, Area2_Bump), with NDT and LFADS as students, and it reports gains: NDT on MC_Maze goes from 0.275 to 0.310 co-bps and from 0.779 to 0.891 vel R². Its teacher is different, though. It sees **behavior (hand velocity) as privileged input**, and the student mixes its masked-modeling loss with an MSE/KL/feature distillation term. Here the teacher is **unsupervised** (spikes only, no behavior), it is **quantized** to 256 prototypes, and its rates *replace* the counts in an otherwise unchanged Poisson loss. The vel R² gain on MC_Maze is similar (+0.06 here, +0.11 there) without behavior ever reaching the teacher.
- **Self-distillation can beat the teacher** in general ML ([Revisiting Self-Distillation, 2022](https://arxiv.org/abs/2206.08491)), so "student beats teacher" is not new in itself. What is reported here is the negative result for *self*-distillation on spikes, and the contrast with a quantized teacher.
- **Distillation failing on Poisson counts:** [SpikeProphecy](https://arxiv.org/abs/2605.12992) reports that KL-on-rates distillation fails for ANN→SNN transfer in spike forecasting (Neuropixels, not NLB).
- **Discrete units as targets:** HuBERT ([Hsu et al. 2021](https://arxiv.org/abs/2106.07447)) is the template for the k-means-on-latents tokenizer. No HuBERT-style target for spiking data was found. The nearest are VQ codebooks in EEG models ([review](https://arxiv.org/abs/2504.20069)) and residual-VQ spike tokens on organoid MEAs ([Tanveer et al. 2026](https://arxiv.org/abs/2609.23907)).

Verdict: **narrow, not open.** BLEND already shows that distilling into NDT lifts co-bps on NLB. What still looks unreported:
- an unsupervised, quantized teacher;
- matched controls showing that generic soft targets (self-distillation, smoothing) don't do it;
- quantization as what makes the teacher's rates a good target.

Placement, for context only (test-phase [EvalAI](https://eval.ai/web/challenges/challenge-page/1256/leaderboard), queried 2026-10-27; val vs test is not like-for-like):
- MC_RTT 5 ms: S5 0.2296, AutoLFADS 0.1868, NDT 0.1749.
- MC_Maze 5 ms: STNDT ensemble 0.3862, NDT 0.3229.

Our 0.1615 (MC_RTT val) sits below test NDT, and our 0.3306 (MC_Maze val) above it. The claim is a relative gain over a matched baseline, not a state-of-the-art number.

## Limits

- Validation split only; all claims are relative, within one pipeline. No test-set or leaderboard placement.
- Absolute numbers stay below acausal state of the art on MC_Maze (STNDT ensemble 0.3862 on the 5 ms test board, per TOKENS.md).
- MC_RTT configs are untuned: vel R² ≈ 0.3 for every model, and the seed spread is wider (0.005 co-bps on `dist_mu_mt`, vel R² 0.33-0.40). Not yet placed against the MC_RTT leaderboard.
- Two-stage training: the student's total budget includes the teacher's autoencoder run; the counts baseline gets no equivalent. Self-distillation is matched on this (two full runs) and still loses.
- Not run on MC_RTT: `dist_ae_mt`, smoothed counts, and the causal students.
- One bin size (5 ms), K = 256 only for the masked-transformer rows.
- **Checkpoint criterion differs** (found in code review, 2026-10-27). The counts / self / smooth runs pick the best epoch by mean heldin NLL + mean heldout NLL. The `dist_*_mt` runs pick by NLL pooled over all 182 neurons, which weights heldout less. This is a confound for the headline gap until the `dist_*_mt` rows are rerun with the same criterion.
- The self-distillation teacher's rates are taken on unmasked input. At heldin positions they may partly track the observed count, which makes this a weaker control than intended. Untested.

## Reproduce

After the README setup. GPU recommended.

```bash
cd baselines
for ds in mc_maze mc_rtt; do
  export NLB_DATASET=$ds  # mc_rtt needs DANDI 000129 under data/000129/sub-Indy
  python prep_tensors_mc_maze.py
  for s in 0 1 2; do
    MT_SEED=$s MT_CONFIG=v4 python run_masked_transformer_mc_maze.py                   # counts
    MT_SEED=$s MT_TARGET=self MT_CONFIG=v4 python run_masked_transformer_mc_maze.py    # self-distillation
    MT_SEED=$s MT_TARGET=smooth MT_CONFIG=v4 python run_masked_transformer_mc_maze.py  # smoothed counts
    VQ_TOKENIZER=aekmeans VQ_K=256 VQ_MODE=dist_mu_mt,dist_ae_mt VQ_SEED=$s \
      python run_vq_ar_transformer_mc_maze.py                                # writes the vq_cont teacher too
  done
  (cd .. && python eval_all_mt.py)
done
```

The `run_*_mc_maze.py` names predate the dataset switch; `NLB_DATASET` selects the data and prefixes the output files.
