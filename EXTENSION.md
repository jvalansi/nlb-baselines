# Beyond the 2021 baselines: transformers on MC_Maze, and a foundation model on FALCON

The four MC_Maze baselines in the [README](README.md) are a 2021-era snapshot: classical co-smoothing (smoothing, GPFA, GRUs). This extension carries the same task forward two steps:

- **Part 1** — a **transformer** on the identical MC_Maze co-smoothing task. It beats every classical baseline on *both* metrics, breaking the smoothness-vs-sharpness trade-off the baseline section identified. A causal variant quantifies the cost of giving up future context — which sets up why the field's live benchmark demands it.
- **Part 2** — **FALCON**, the live benchmark for *streaming, cross-session* intracortical decoding, and what a modern neural foundation model (NDT3) does on it, honestly scored. The short version: on FALCON H1's public test leaderboard our fine-tune placed **~5th of 13** at submission (cross-day held-out R² 0.556; 8th of 18 as of 2026-10-05 as newer entries arrived), just below NDT3's own entry — and the untouched pretrained checkpoint scores within noise of it (0.548), so on this dataset fine-tuning does not robustly beat the pretrained baseline. The contribution there is a working end-to-end fine-tune + local-scoring + streaming-decode pipeline, six decoder bug-fixes that surfaced only when the decode path was actually exercised, and an honest read on what does and doesn't move the metric.

---

# Part 1 — Transformers on MC_Maze

Same task as the four baselines in the README: given spikes from the 137 held-in neurons, predict firing rates for all 182 neurons over each trial's 140 bins (5 ms). Same val split, same `nlb_tools.evaluation.evaluate`. Reproducible in this repo — see [Reproducibility](#reproducibility).

## Masked transformer

**Architecture** (`baselines/run_masked_transformer_mc_maze.py`). Per-neuron read-in `E_in[137, d]` (`h = spikes @ E_in`), learned positional embeddings over the 140 bins, a stack of pre-norm transformer encoder layers with **full bidirectional attention**, linear read-out `E_out[d, 182]` + softplus → per-(neuron, bin) rates. Training: random (t, n) entries of the held-in input are masked at some rate; Poisson NLL on the masked held-in entries plus Poisson NLL on all held-out entries (the co-smoothing target, never masked). AdamW, cosine LR 3e-4→3e-5, 500 epochs, best-co-bps checkpoint. ~1M params. GRAFT-simplified: session-specific, no cross-session unit embeddings.

**Sweep** (MC_Maze val, 5 ms bins):

| Config | co-bps | vel R² |
|---|---:|---:|
| v1 (4L, d=128, 4 heads, ffn=512, mask 0.25) | 0.3100 | 0.8123 |
| v2 (v1, mask 0.40) | 0.3019 | 0.8164 |
| v3 (v1, d=192) | 0.3139 | 0.8117 |
| **v4 (v1, 6 layers)** | **0.3142** | **0.8235** |
| v5 (6L + d=192 + mask 0.40) | 0.2937 | 0.8178 |

Reading the sweep: **depth helps** (v4's 6 layers win), **more width alone doesn't** (v3), **higher mask rate hurts co-bps** (v2, v5), and **stacking changes overfits** (v5 is worst despite being biggest). v4 is the winner on both metrics.

## Autoregressive (causal) variant

**Architecture** (`baselines/run_ar_transformer_mc_maze.py`). Same read-in / read-out / positional scheme, but a **causal attention mask**, input shifted right, and a next-timestep Poisson NLL objective — the model predicts bin *t*'s rates from bins < *t* only. No future context, no masking. This is the streaming-decoder shape: it's the objective FALCON (Part 2) actually requires.

Result: **co-bps 0.2712 / vel R² 0.7826.**

## What Part 1 shows

Against the four classical baselines (best-in-class: GRU-v1 co-bps 0.2328; GPFA vel R² 0.6403):

- **The masked transformer wins both metrics decisively.** v4's 0.3142 co-bps is **+35% over GRU-v1**; its 0.8235 vel R² is far above GPFA's 0.6403. The README's baseline section framed co-bps (sharpness) and vel R² (smoothness) as a trade-off no single classical model won — the transformer wins both at once. Attention models neuron-specific temporal structure *and* clean population trajectories together.
- **The causal variant costs ~14% co-bps** (0.2712 vs 0.3142) — the price of giving up future context — **but still beats every classical baseline on both metrics**, GRU-v1 included. Causal decoding from spikes is viable, not a large sacrifice. That matters because the field's live benchmark is causal by construction, which is Part 2.

---

# Part 2 — From NLB'21 to FALCON

The masked transformer above is still the NLB'21 world: batch, non-causal, single session, offline. **FALCON** is where the field went to test what deployment actually needs.

> **Scope note.** Part 2 is a write-up of results obtained with a separate NDT3 + FALCON pipeline (vendored NDT3, GPU fine-tuning, Docker-based EvalAI submission). That pipeline is **not** included in this repo — it's a heavy, benchmark-specific setup. The methodology is documented here in enough detail to reproduce against [`joel99/ndt3`](https://huggingface.co/joel99/ndt3) and the official [`falcon_challenge`](https://github.com/snel-repo/falcon-challenge) evaluator; the transformer code in Part 1 is what this repo actually ships.

## How the field moved: NLB'21 → FALCON

- **NLB'21** (co-smoothing on MC_Maze, as in the README): predict held-out *neuron rates* from held-in neurons, whole trial visible at once (non-causal), single session, one monkey. As of **January 2026 the EvalAI challenge no longer accepts submissions** — the organizers can no longer host it, so no new model can appear on the public leaderboard ([NLB challenge guidelines](https://neurallatents.github.io/challenge.html)).
- **FALCON** — *Few-shot Algorithms for Consistent Neural Decoding* ([Karpowicz et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf); [bioRxiv 2024.09.15.613126](https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1)), EvalAI challenge 2319. Predict *behavior* from spikes under **causal streaming inference** (no future context), across **multiple session-days**, with a **few-shot recalibration** protocol targeting the nonstationarity that forces real iBCIs to recalibrate. Five datasets: **H1** human reach & grasp, **H2** human handwriting, **M1** monkey reach & grasp, **M2** monkey finger movement, **B1** birdsong.

The axis that matters for real iBCI: FALCON scores **held-out** (cross-day) sessions, because the deployment problem is a decoder that keeps working on a day it was never calibrated on. That is the number the leaderboard ranks, and — after a months-long EvalAI outage — the number we finally obtained (see [The held-out placement](#the-held-out-placement)).

## The model: NDT3

**NDT3** — Neural Data Transformer 3, published as *A Generalist Intracortical Motor Decoder* ([Ye et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/a00000e6a2208172700510bcd69d48e9-Paper-Conference.pdf); [bioRxiv 2025.02.02.634313](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1)) — is a multimodal transformer pretrained on ~2000 hours of intracortical spiking + motor covariates from 30+ monkeys and humans across 10 labs, tokenizing spikes in 20 ms bins and channel patches (~32). Its [`joel99/ndt3` HuggingFace repo](https://huggingface.co/joel99/ndt3) is, as of this writing, the only iBCI foundation model with *real, usable* pretrained checkpoints. [GRAFT](https://arxiv.org/abs/2606.11066) (Ge & Xie, 2026) has a paper reporting SOTA MC_Maze co-bps (0.3866 ensemble) but no public code or released checkpoints we could find; the [`NerDSLab/POYO`](https://huggingface.co/NerDSLab/POYO) / POYO+ repos are placeholders (only `.gitattributes`, no weights). So NDT3 is the only one you can actually load and run.

Checkpoints come in two variants: `base_45m_1kh` (45M params, ~1k hours of pretraining) and `big_350m_2kh` (350M params, ~2k hours). The repo also distinguishes **raw backbones** (indexed by pretraining run-id) from **task fine-tunes** (published per FALCON dataset under `h1/`, `m1/`, `m2/`).

## Results: FALCON H1

Two metrics, and the gap between them is the whole story.

- **our val R²** — `val_kinematic_r2` on *our own* held-in carve-out of the calibration data. **Selection-biased**: we tuned the learning rate against this split, so it flatters whatever we selected on it.
- **minival Held-In R²** — FALCON's canonical held-in metric, computed by the official evaluator (`--evaluation local --phase minival`). Comparable across checkpoints; "easy" because held-in = the same session-days seen in calibration.

| Model | init | our val R² (selection-biased) | minival Held-In R² |
|---|---|---:|---:|
| `base_45m_1kh` | pretrained (untouched HF ckpt) | 0.6025 | **0.8907 ± 0.044** |
| `base_45m_1kh` | our LR=1e-5 fine-tune | 0.6082 | 0.8901 ± 0.050 |
| `big_350m_200h` | FALCON fine-tune | 0.5892 | — |
| `big_350m_2kh` | raw backbone (`900t21lf`) | 0.5403 | — |

Three readings, all load-bearing:

1. **The 0.890 is real, not a patch artifact.** Two independently-derived checkpoints (untouched pretrained vs our fine-tune) land within **0.0006** of each other on the canonical held-in split. That agreement is what validates the decode pipeline — the patched decoder is producing a genuine number.
2. **Fine-tuning does not beat pretrained on H1.** The fine-tune bought +0.0057 on *our own* carve-out (which we tuned LR against) and **−0.0006** on FALCON's canonical split. The LR sweep selected for our split, not for generalization. The honest headline is *"pretrained NDT3 is already at its H1 ceiling,"* not *"we beat the baseline."*
3. **Bigger is worse here.** Both 350M variants overfit H1's ~170 held-in trials (train R² ~0.91 vs val ~0.54). Capacity-vs-data mismatch is the binding constraint, not pretraining budget — so `base_45m_1kh` stays the entry.

### The LR sweep (why default fine-tuning underperformed)

NDT3's default fine-tune LR (4e-4) overshoots the pretrained optimum on a small calibration split. Lowering it recovers the pretrained level on our val carve-out but, as (2) shows, doesn't generalize past it:

| LR | our val R² |
|---|---:|
| 4e-4 (default) | 0.5684 |
| 1e-4 | 0.5835 |
| 4e-5 | 0.5995 |
| **1e-5** | **0.6082** |

## The held-out placement

Every number above is **held-in** — it answers "does the decoder work on a session-day it was calibrated on." The number that *ranks* the leaderboard is **held-out**: cross-day sessions whose data files are server-side only, reachable only by submitting to EvalAI and being scored. For most of this work the challenge's evaluation queue was not scoring our submissions at all ([snel-repo/falcon-challenge#32](https://github.com/snel-repo/falcon-challenge/issues/32)), so we validated locally on held-in data. When the queue came back online, both entries scored on the H1 test phase:

| Model (H1 test phase) | Held-Out R² (ranked) | Held-In R² | norm. latency |
|---|---:|---:|---:|
| `base_45m_1kh`, our LR=1e-5 fine-tune (public) | **0.556 ± 0.090** | 0.660 ± 0.025 | 0.075 |
| `base_45m_1kh`, pretrained (untouched HF ckpt) | 0.548 ± 0.094 | 0.670 ± 0.025 | — |
| official `ndt3` team entry | 0.574 | — | — |

The fine-tune ranked **~5th of 13** on the public H1 test board at submission (8th of 18 as of 2026-10-05) — clearing SPINT, Credasis AI, and most FALCON baseline variants, sitting just below NDT3's own entry. (Note these test-phase held-in numbers, ~0.66, are a *different split* from the minival held-in ~0.89 above and are not comparable to it.)

Two readings, both consistent with the held-in story:

1. **Fine-tuning still does not robustly beat pretrained.** The fine-tune edges the untouched checkpoint by +0.008 held-out (0.556 vs 0.548) — the *opposite* direction from minival (−0.0006), and well inside the ±0.09 std. Across both splits the honest read is a wash: on H1, pretrained NDT3 is already at its ceiling.
2. **The ~0.02 gap to NDT3's own entry is not our fine-tune underperforming.** The untouched pretrained checkpoint *also* trails the official `ndt3` entry (0.548 vs 0.574), so that entry used a stronger config or checkpoint than the public `base_45m_1kh` — it is not a deficit introduced by our fine-tuning.

So the honest boundary, now closed: the held-in 0.890 validated the decode pipeline; the held-out 0.556 is the deployment-relevant, leaderboard-ranked placement — and it confirms rather than overturns the held-in read.

## What actually took the work: the decode path

Fine-tuning NDT3 was mostly config archaeology. The real engineering was **making the streaming decoder run at all** — and it hadn't, because the only thing that had ever exercised it was stuck EvalAI submissions that never executed server-side. In this vendored snapshot, NDT3's `__init__` in `context_general_bci/ndt3_slim.py` had drifted out of sync with the inference functions that call it. Running the official evaluator locally surfaced the whole cluster at once:

1. **A constructor positional-argument collision.** `from_training_shell` passed `reward_task` and `reward_quantizer` *positionally* into an `__init__` that no longer declares them. Every positional after that point shifts by two — so `max_spatial_position` silently lands in the `neurons_per_token` slot, mis-sizing the model. Fix: drop those two arguments from the call.
2. **Five attributes `predict_prefill` reads but `__init__` never sets** — `neural_task`, `return_task`, `reward_quantizer`, `reward_task`, `split_return_reward` (plus `reward_return_pad_value`, set defensively). The decode loop would `AttributeError` on the first real streaming call. Fix: initialize them in `__init__`.

None of these could surface until the decode path actually ran end to end — which, before local scoring, it never had. This is the transferable artifact: a foundation-model streaming decoder that has been exercised on the canonical evaluator, not just built.

## Gotchas (FALCON / NDT3 specifics)

Continuing the README's gotchas section, for the next person fine-tuning NDT3:

1. **Deep Learning AMI CUDA 12.8 vs NDT3's torch 2.1+cu118.** Pin torch to the cu118 build; use the DLAMI's built-in `pytorch` venv rather than fighting the system CUDA.
2. **numpy 2 vs numpy 1, `transformers>=2.4` vs 2.1.** NDT3 wants the older pins; a fresh install pulls incompatible majors.
3. **Hydra chokes on `=` in checkpoint paths.** Config-override paths with `=` in them need escaping.
4. **`SpikingDataset` preprocess has a check-then-write race.** 16 parallel workers race on `falcon_h1_norm.pth`; readers hit truncated files (`PytorchStreamReader failed reading file byteorder`). Serialize with `load_workers=1`.
5. **The raw `big_350m_2kh` backbone stores un-anonymized subject IDs** (Pitt `CRS02b/07/08`) — fine-tuning it needs a `SubjectName` patch. And per NDT3's own dockerfile, **do not submit the 2kh model — it bleeds the test individual.**
6. **docker-py 7 breaks the EvalAI submit client** (`docker_image_size` is `None`; the push stream no longer emits an `aux` dict) — both need in-place patches to `evalai/submissions.py` if you script submissions.

## Lessons (continuing from the README's set)

3. **A foundation model can be at its ceiling before you touch it.** On a small held-in calibration set, a well-pretrained checkpoint may already saturate the metric; a fine-tune "win" on your own carve-out can be pure selection noise. Always score the untouched checkpoint through the identical evaluator as your standing baseline — a fine-tune number without it is meaningless.
4. **Bigger overfits small calibration sets.** 45M beat 350M on H1's ~170 held-in trials. Match capacity to calibration data, not to pretraining budget.
5. **Hold the honest held-in number until the held-out one lands — then report both.** The held-in 0.890 is honest and reproducible but is *not* the deployment metric (cross-day held-out); reporting it alone would have overstated the result. The held-out placement (0.556, ~5th/13) is the one that ranks — and it confirmed the held-in read: fine-tuning doesn't beat pretrained on H1. Always be explicit about which split every number is on.

## Reproducibility

**Part 1 (transformers on MC_Maze) is fully reproducible in this repo.** After the dataset download in the README:

```bash
# one-time: bake train/eval tensors to data/mc_maze_5ms.npz (avoids reloading the NWB per config)
python baselines/prep_tensors_mc_maze.py

# masked-transformer sweep (writes outputs/mc_maze_mt_<config>_output_val.h5)
MT_CONFIG=v1 python baselines/run_masked_transformer_mc_maze.py
MT_CONFIG=v2 python baselines/run_masked_transformer_mc_maze.py
MT_CONFIG=v3 python baselines/run_masked_transformer_mc_maze.py
MT_CONFIG=v4 python baselines/run_masked_transformer_mc_maze.py   # winner
MT_CONFIG=v5 python baselines/run_masked_transformer_mc_maze.py

# causal (autoregressive) variant
python baselines/run_ar_transformer_mc_maze.py

# score every transformer output against the MC_Maze val targets
python eval_all_mt.py
```

(Config names in the sweep table map to the `MT_CONFIG` env var; check the top of `run_masked_transformer_mc_maze.py` for the exact hyperparameters each selects.)

**Part 2 (FALCON/NDT3) is not vendored here** — see the scope note above. To reproduce: fine-tune a `base_45m_1kh` checkpoint from [`joel99/ndt3`](https://huggingface.co/joel99/ndt3) on FALCON H1 at LR 1e-5, and score it with the official [`falcon_challenge`](https://github.com/snel-repo/falcon-challenge) evaluator in local mode (`--evaluation local --phase minival`) for the held-in number, then submit to the EvalAI `test` phase for the ranked held-out placement. Always score the untouched pretrained checkpoint through the same evaluator as your baseline (Lesson 3).

## Citation

Alongside the existing NLB'21 citation:

> Karpowicz, B., Ye, J., Fan, C., et al. (Pandarinath, C., senior author). *Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.* NeurIPS 2024 Datasets and Benchmarks Track. https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1

> Ye, J., Rizzoglio, F., et al. *A Generalist Intracortical Motor Decoder* (NDT3). NeurIPS 2025. https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1
