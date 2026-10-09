"""Codebook (VQ) spike tokens + autoregressive transformer for MC_Maze val.

Question: what does forcing each 5 ms population state through one of K
prototypes cost on co-bps / vel R², against the continuous AR transformer
(run_ar_transformer_mc_maze.py, 0.2712 / 0.7826)? Same backbone, same
hyperparameters, same val split — only the bottleneck changes.

Tokenizer (VQ_TOKENIZER):
 kmeans (no training):
  feature_t = per-neuron z-scored causal EMA (tau VQ_TAU ms) of heldin counts
  z_t       = nearest of K centroids (k-means on all train bins)
  mu[k]     = mean Hi+Ho counts of train bins with z_t = k, shrunk toward the
              global mean (pseudo-count ALPHA) so no prototype has a zero rate
 aekmeans (learned features, discrete units as in HuBERT / GSLM):
  the vqvae network below trained with no bottleneck, then k-means (K) on its
  causal latents; mu[k] = shrunk empirical mean counts, as for kmeans.
 vqvae (learned):
  causal transformer encoder over heldin counts (same backbone, unshifted, so
  z_t sees heldin <= t) -> EMA vector quantizer (K codes) -> MLP decoder ->
  Hi+Ho rates. Trained on Poisson NLL with coordinated dropout (the first
  version, without it, learned to copy the current heldin bin: co-bps ~0), so codes are chosen to be predictive of
  heldout neurons rather than to cluster heldin. mu[k] = decoder(code_k).
  First VQVAE_WARMUP_FRAC of epochs run unquantized; the codebook is then
  k-means-initialized on the latents (random init collapsed to one code).

Modes (VQ_MODE, comma-separated; one h5 per mode x K):
  lookup  rates_t = mu[z_t]. No model. Ceiling on what the codebook can say
          about heldout neurons given the current heldin state.
  out     raw heldin counts in (shift-right, causal) -> logits over K;
          rates_t = softmax(logits_t) @ mu. Quantized output only.
  inout   one-hot tokens in (shift-right; the zero vector at t=0 acts as BOS)
          -> logits over K. Pure spike-token LM: quantized input and output.
  out_pois / inout_pois
          CE-vs-Poisson ablation: same input, network and mixture head
          softmax(logits_t) @ mu, but trained on Poisson NLL of those rates
          against the counts instead of CE on the tokenizer's next token.
  dist_ae / dist_mu (aekmeans only)
          Distillation control for `out`: the continuous AR model (softplus head
          over Hi+Ho, no tokens) trained on Poisson NLL against soft targets -
          the unquantized AE's rates at t (dist_ae) or the token prototype
          mu[z_t] (dist_mu) - instead of the counts. If dist_ae matches `out`,
          the gain is distillation from the AE, not CE on discrete tokens.

Training (out / inout): cross-entropy on the next token. Checkpoint selected by
val Poisson NLL of the mixture rates (the quantity co-bps scores).
"""
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from run_ar_transformer_mc_maze import (
    CONFIGS, ARTransformer, poisson_nll_elementwise, shift_right,
)

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
PHASE = "val"
OUTPUT_KEY = DATASET_NAME
VAL_SPLIT = 0.1
SEED = 0  # val split
TORCH_SEED = int(os.getenv("VQ_SEED", "0"))  # init, k-means, batch order; split stays fixed

CONFIG = os.getenv("VQ_CONFIG", "v1")
C = CONFIGS[CONFIG]
KS = [int(k) for k in os.getenv("VQ_K", "64,256,1024").split(",")]
MODES = os.getenv("VQ_MODE", "lookup,out,inout").split(",")
TAU_MS = float(os.getenv("VQ_TAU", "25"))
ALPHA = float(os.getenv("VQ_ALPHA", "10"))
KMEANS_ITERS = 50
TOKENIZER = os.getenv("VQ_TOKENIZER", "kmeans")
VQVAE_EPOCHS = int(os.getenv("VQVAE_EPOCHS", "400"))
VQVAE_LR = C["lr"]  # 1e-3 stalled at the null heldout NLL; the AR baseline's lr learns
CD_P = 0.3  # coordinated-dropout rate on heldin inputs
CODE_DIM = 32  # low-dim lookup space
VQVAE_WARMUP_FRAC = 0.5  # epochs as a plain autoencoder before the codebook is k-means-initialized
COMMIT_BETA = 0.25
EMA_DECAY = 0.99
DEAD_HITS = 1.0  # EMA hits/batch below which a code is restarted from a random encoder output

DIR = Path(__file__).resolve().parents[1] / "outputs"
DIR.mkdir(exist_ok=True)
NPZPATH = DIR.parent / "data" / f"{DATASET_NAME}_{BIN_SIZE_MS}ms.npz"


def causal_ema(x, tau_ms):
    """x (N, T, n) counts -> exponential moving average over T, causal."""
    a = 1.0 - math.exp(-BIN_SIZE_MS / tau_ms)
    out = np.empty_like(x)
    acc = np.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        acc = (1 - a) * acc + a * x[:, t]
        out[:, t] = acc
    return out


def kmeans(feats, K, iters, device, seed=TORCH_SEED):
    """feats (M, d) float tensor -> centroids (K, d). Empty clusters re-seeded from data."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = feats[torch.randperm(len(feats), generator=g)[:K]].clone()
    for _ in range(iters):
        z = assign(feats, cent)
        sums = torch.zeros_like(cent).index_add_(0, z, feats)
        cnt = torch.bincount(z, minlength=K).float()
        empty = cnt == 0
        cent = sums / cnt.clamp(min=1).unsqueeze(1)
        if empty.any():
            cent[empty] = feats[torch.randint(len(feats), (int(empty.sum()),), generator=g)].to(feats.device)
    return cent


def assign(feats, cent, chunk=65536):
    return torch.cat([torch.cdist(feats[i:i + chunk], cent).argmin(1) for i in range(0, len(feats), chunk)])


def prototype_rates(z, counts, K, alpha):
    """Shrunk mean counts per token. z (M,), counts (M, n) -> (K, n). Pass train-split bins only."""
    sums = torch.zeros(K, counts.shape[1], device=counts.device).index_add_(0, z, counts)
    cnt = torch.bincount(z, minlength=K).float().unsqueeze(1)
    prior = counts.mean(0, keepdim=True)
    return (sums + alpha * prior) / (cnt + alpha)


def cosine_schedule(opt, total_steps):
    def lr_mul(step):
        if step < C["warmup"]:
            return step / max(1, C["warmup"])
        p = min(1.0, (step - C["warmup"]) / max(1, total_steps - C["warmup"]))
        floor = C["min_lr"] / C["lr"]
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_mul)


class EMAQuantizer(nn.Module):
    """VQ-VAE codebook with EMA updates (van den Oord 2017, App. A.1), k-means init after an
    unquantized warmup, and restarts for codes averaging <1 hit per batch."""

    def __init__(self, K, d):
        super().__init__()
        self.K = K
        self.register_buffer("embed", torch.zeros(K, d))
        self.register_buffer("size", torch.zeros(K))
        self.register_buffer("embed_sum", torch.zeros(K, d))
        self.register_buffer("initted", torch.tensor(False))

    def codes(self, z_e):
        return assign(z_e, self.embed)

    @torch.no_grad()
    def init_from(self, z_e):
        """k-means init on encoder outputs from a warmed-up (unquantized) encoder."""
        self.embed.copy_(kmeans(z_e, self.K, KMEANS_ITERS, z_e.device))
        self.size.fill_(2 * DEAD_HITS)  # unused codes restart within ~70 steps
        self.embed_sum.copy_(self.embed * 2 * DEAD_HITS)
        self.initted.fill_(True)

    def forward(self, z_e):
        if not self.initted:  # warmup phase: plain autoencoder, no bottleneck
            return z_e, None, z_e.new_zeros(())
        idx = self.codes(z_e)
        z_q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = nn.functional.one_hot(idx, self.K).type_as(z_e)
                self.size.mul_(EMA_DECAY).add_(onehot.sum(0), alpha=1 - EMA_DECAY)
                self.embed_sum.mul_(EMA_DECAY).add_(onehot.t() @ z_e, alpha=1 - EMA_DECAY)
                n = self.size.sum()
                size = (self.size + 1e-5) / (n + self.K * 1e-5) * n
                self.embed.copy_(self.embed_sum / size.unsqueeze(1))
                dead = self.size < DEAD_HITS
                if dead.any():
                    fresh = z_e[torch.randint(len(z_e), (int(dead.sum()),), device=z_e.device)]
                    self.embed[dead] = fresh
                    self.embed_sum[dead] = fresh
                    self.size[dead] = DEAD_HITS
        commit = COMMIT_BETA * nn.functional.mse_loss(z_e, z_q.detach())
        return z_e + (z_q - z_e).detach(), idx, commit


class VQVAE(nn.Module):
    def __init__(self, n_heldin, n_out, T, K):
        super().__init__()
        d = C["d_model"]
        self.enc = ARTransformer(n_heldin, CODE_DIM, T, d, C["n_layers"], C["n_heads"], C["ffn"], C["dropout"])
        self.vq = EMAQuantizer(K, CODE_DIM)
        self.dec = nn.Sequential(nn.Linear(CODE_DIM, 256), nn.GELU(), nn.Linear(256, n_out))

    def encode(self, x):
        # causal, unshifted: z_t sees heldin <= t
        return logits_of(self.enc, x)

    def forward(self, x):
        z_e = self.encode(x)
        B, T, d = z_e.shape
        z_q, idx, commit = self.vq(z_e.reshape(-1, d))
        rates = nn.functional.softplus(self.dec(z_q)).reshape(B, T, -1)
        return rates, None if idx is None else idx.reshape(B, T), commit

    @torch.no_grad()
    def tokens(self, x, chunk=256):
        return torch.cat([self.vq.codes(self.encode(x[i:i + chunk]).reshape(-1, self.vq.embed.shape[1]))
                          .reshape(len(x[i:i + chunk]), -1) for i in range(0, len(x), chunk)])

    @torch.no_grad()
    def prototype_rates(self):
        return nn.functional.softplus(self.dec(self.vq.embed))


def train_vqvae(K, xt, yt, tr_idx, val_idx, device, quantize=True):
    """Fit tokenizer; return (model, mu). Checkpoint by val heldout Poisson NLL of decoded rates.
    quantize=False: plain causal autoencoder throughout (for aekmeans); mu is None."""
    torch.manual_seed(TORCH_SEED)
    n_train, T, n_heldin = xt.shape
    model = VQVAE(n_heldin, yt.shape[2], T, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=VQVAE_LR, weight_decay=C["wd"])
    bs = C["batch"]
    n_batches = int(np.ceil(len(tr_idx) / bs))
    sched = cosine_schedule(opt, n_batches * VQVAE_EPOCHS)
    warm = max(1, int(VQVAE_EPOCHS * VQVAE_WARMUP_FRAC)) if quantize else VQVAE_EPOCHS + 1
    best, best_state = float("inf"), None
    for epoch in range(VQVAE_EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        tl = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            # Coordinated dropout (Keshtkaran & Pandarinath 2019): score heldin only where it was
            # masked from the input, so the encoder can't copy the current bin into the code.
            drop = torch.rand(xt[idx].shape, device=device) < CD_P
            rates, _, commit = model(xt[idx] * ~drop / (1 - CD_P))
            w = torch.cat([drop, torch.ones_like(yt[idx][:, :, n_heldin:], dtype=torch.bool)], 2).float()
            loss = (poisson_nll_elementwise(rates, yt[idx]) * w).sum() / w.sum() + commit
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            rates, z, _ = model(xt[val_idx])
            # heldout neurons only: heldin at t is in the input, so its NLL would reward copying
            val_nll = poisson_nll_elementwise(rates[:, :, n_heldin:], yt[val_idx][:, :, n_heldin:]).mean().item()
            used = int(torch.unique(z).numel()) if z is not None else 0
        if epoch + 1 == warm:
            with torch.no_grad():
                model.eval()
                sample = torch.cat([model.encode(xt[tr_idx[i:i + 256]]).reshape(-1, model.vq.embed.shape[1])
                                    for i in range(0, len(tr_idx), 256)])
                model.vq.init_from(sample)
            best = float("inf")  # only quantized checkpoints are eligible
            print(f"[{TOKENIZER} K={K}] warmup done at epoch {epoch}; codebook k-means-initialized", flush=True)
            continue
        if val_nll < best:
            best = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == VQVAE_EPOCHS - 1:
            print(f"[{TOKENIZER} K={K}] epoch {epoch:3d}  train {tl/len(tr_idx):.4f}  val_nll {val_nll:.4f}  "
                  f"best {best:.4f}  codes_used(val) {used}/{K}  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, (model.prototype_rates() if quantize else None)


def logits_of(model, x):
    """ARTransformer forward without the softplus — read_out gives token logits."""
    h = model.read_in(x) + model.pos
    h = model.encoder(h, mask=model.causal_mask, is_causal=True)
    return model.read_out(h)


def write_h5(path, train_rates, eval_rates, n_heldin):
    with h5py.File(path, "w") as f:
        g = f.create_group(OUTPUT_KEY)
        g.create_dataset("train_rates_heldin", data=train_rates[:, :, :n_heldin])
        g.create_dataset("train_rates_heldout", data=train_rates[:, :, n_heldin:])
        g.create_dataset("eval_rates_heldin", data=eval_rates[:, :, :n_heldin])
        g.create_dataset("eval_rates_heldout", data=eval_rates[:, :, n_heldin:])
    print(f"Wrote {path}")


def train_lm(mode, K, x_in_all, z_all, mu, y_all, tr_idx, val_idx, x_in_eval, device):
    """Train next-token LM; return (train_rates, eval_rates) as numpy."""
    torch.manual_seed(TORCH_SEED)
    n_in = x_in_all.shape[2]
    T = x_in_all.shape[1]
    model = ARTransformer(n_in, K, T, C["d_model"], C["n_layers"], C["n_heads"],
                          C["ffn"], C["dropout"]).to(device)
    print(f"[{mode} K={K}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    x_tr, z_tr, y_tr = x_in_all[tr_idx], z_all[tr_idx], y_all[tr_idx]
    x_val, z_val, y_val = x_in_all[val_idx], z_all[val_idx], y_all[val_idx]
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=C["wd"])
    bs, epochs = C["batch"], C["epochs"]
    n_batches = int(np.ceil(len(tr_idx) / bs))
    total = n_batches * epochs

    sched = cosine_schedule(opt, total)
    ce = nn.CrossEntropyLoss()
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx), device=device)
        tl = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            logits = logits_of(model, shift_right(x_tr[idx]))
            if mode.endswith("_pois"):
                loss = poisson_nll_elementwise(logits.softmax(-1) @ mu, y_tr[idx]).mean()
            else:
                loss = ce(logits.reshape(-1, K), z_tr[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv = logits_of(model, shift_right(x_val))
            val_ce = ce(lv.reshape(-1, K), z_val.reshape(-1)).item()
            val_nll = poisson_nll_elementwise(lv.softmax(-1) @ mu, y_val).mean().item()
        if val_nll < best:
            best = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{mode} K={K}] epoch {epoch:3d}  train_loss {tl/len(tr_idx):.4f}  val_ce {val_ce:.4f}  "
                  f"val_nll {val_nll:.4f}  best {best:.4f}  lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr = (logits_of(model, shift_right(x_in_all)).softmax(-1) @ mu).cpu().numpy()
        ev = (logits_of(model, shift_right(x_in_eval)).softmax(-1) @ mu).cpu().numpy()
    return tr, ev


def train_student(mode, x_all, target, y_all, tr_idx, val_idx, x_eval, device):
    """Continuous AR model fit to soft targets (Poisson NLL vs teacher rates); checkpoint by
    val Poisson NLL against the real counts, as for every other model. Returns numpy rates."""
    torch.manual_seed(TORCH_SEED)
    n_out, T = y_all.shape[2], x_all.shape[1]
    model = ARTransformer(x_all.shape[2], n_out, T, C["d_model"], C["n_layers"], C["n_heads"],
                          C["ffn"], C["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=C["wd"])
    bs, epochs = C["batch"], C["epochs"]
    n_batches = int(np.ceil(len(tr_idx) / bs))
    sched = cosine_schedule(opt, n_batches * epochs)
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        tl = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            loss = poisson_nll_elementwise(model(shift_right(x_all[idx])), target[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_nll = poisson_nll_elementwise(model(shift_right(x_val)), y_val).mean().item()
        if val_nll < best:
            best = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{mode}] epoch {epoch:3d}  train_loss {tl/len(tr_idx):.4f}  val_nll {val_nll:.4f}  "
                  f"best {best:.4f}  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(shift_right(x_all)).cpu().numpy(), model(shift_right(x_eval)).cpu().numpy()


def main():
    np.random.seed(SEED)
    torch.manual_seed(TORCH_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  config={CONFIG}  tokenizer={TOKENIZER}  K={KS}  modes={MODES}  tau={TAU_MS}ms  alpha={ALPHA}")

    npz = np.load(NPZPATH)
    x_train = npz["train_heldin"].astype(np.float32)
    y_train_ho = npz["train_heldout"].astype(np.float32)
    x_eval = npz["eval_heldin"].astype(np.float32)
    n_train, T, n_heldin = x_train.shape
    y_train = np.concatenate([x_train, y_train_ho], axis=2)

    # Same val split as the AR baseline (same seed, same call order).
    perm = np.random.permutation(n_train)
    n_val = int(round(n_train * VAL_SPLIT))
    val_idx, tr_idx = torch.from_numpy(perm[:n_val]).to(device), torch.from_numpy(perm[n_val:]).to(device)

    f_train = causal_ema(x_train, TAU_MS)
    f_eval = causal_ema(x_eval, TAU_MS)
    m, s = f_train.reshape(-1, n_heldin).mean(0), f_train.reshape(-1, n_heldin).std(0) + 1e-6
    f_train = torch.from_numpy((f_train - m) / s).to(device)
    f_eval = torch.from_numpy((f_eval - m) / s).to(device)

    xt = torch.from_numpy(x_train).to(device)
    xe = torch.from_numpy(x_eval).to(device)
    yt = torch.from_numpy(y_train).to(device)

    for K in KS:
        if TOKENIZER == "kmeans":
            cent = kmeans(f_train.reshape(-1, n_heldin), K, KMEANS_ITERS, device)
            z_train = assign(f_train.reshape(-1, n_heldin), cent).reshape(n_train, T)
            z_eval = assign(f_eval.reshape(-1, n_heldin), cent).reshape(len(x_eval), T)
            mu = prototype_rates(z_train[tr_idx].reshape(-1), yt[tr_idx].reshape(-1, yt.shape[2]), K, ALPHA)  # tr_idx only: val trials must not leak into mu
        elif TOKENIZER == "vqvae":
            tok, mu = train_vqvae(K, xt, yt, tr_idx, val_idx, device)
            z_train, z_eval = tok.tokens(xt), tok.tokens(xe)
            del tok
        elif TOKENIZER == "aekmeans":
            ae, _ = train_vqvae(K, xt, yt, tr_idx, val_idx, device, quantize=False)
            with torch.no_grad():
                h_tr = torch.cat([ae.encode(xt[i:i + 256]) for i in range(0, n_train, 256)]).reshape(-1, CODE_DIM)
                h_ev = torch.cat([ae.encode(xe[i:i + 256]) for i in range(0, len(xe), 256)]).reshape(-1, CODE_DIM)
                # Control: the same network, unquantized - the ceiling for its tokens.
                cont = {n: torch.cat([nn.functional.softplus(ae.dec(ae.encode(xx[i:i + 256])))
                                      for i in range(0, len(xx), 256)])
                        for n, xx in (("tr", xt), ("ev", xe))}
            teacher = cont["tr"]  # dist_ae target
            cont = {n: v.cpu().numpy() for n, v in cont.items()}
            write_h5(DIR / f"{OUTPUT_KEY}_vq_cont_k{K}_aekmeans{'' if CONFIG == 'v1' else '_' + CONFIG}{'' if TORCH_SEED == 0 else f'_s{TORCH_SEED}'}_output_{PHASE}.h5", cont["tr"], cont["ev"], n_heldin)
            del ae, cont
            cent = kmeans(h_tr, K, KMEANS_ITERS, device)
            z_train = assign(h_tr, cent).reshape(n_train, T)
            z_eval = assign(h_ev, cent).reshape(len(x_eval), T)
            mu = prototype_rates(z_train[tr_idx].reshape(-1), yt[tr_idx].reshape(-1, yt.shape[2]), K, ALPHA)  # tr_idx only: val trials must not leak into mu
        else:
            raise ValueError(f"unknown VQ_TOKENIZER {TOKENIZER}")
        used = torch.bincount(z_train.reshape(-1), minlength=K)
        print(f"[K={K}] tokens used {int((used > 0).sum())}/{K}  "
              f"perplexity(unigram) {math.exp(-(used/used.sum()).clamp(min=1e-12).log().mul(used/used.sum()).sum().item()):.1f}")

        for mode in MODES:
            tag = (f"vq_{mode}_k{K}" + ("" if TOKENIZER == "kmeans" else f"_{TOKENIZER}")
                   + ("" if CONFIG == "v1" else f"_{CONFIG}") + ("" if TORCH_SEED == 0 else f"_s{TORCH_SEED}"))
            path = DIR / f"{OUTPUT_KEY}_{tag}_output_{PHASE}.h5"
            if mode == "lookup":
                tr, ev = mu[z_train].cpu().numpy(), mu[z_eval].cpu().numpy()
            elif mode in ("out", "out_pois"):
                tr, ev = train_lm(mode, K, xt, z_train, mu, yt, tr_idx, val_idx, xe, device)
            elif mode in ("inout", "inout_pois"):
                oh_tr = nn.functional.one_hot(z_train, K).float()
                oh_ev = nn.functional.one_hot(z_eval, K).float()
                tr, ev = train_lm(mode, K, oh_tr, z_train, mu, yt, tr_idx, val_idx, oh_ev, device)
            elif mode in ("dist_ae", "dist_mu"):
                if TOKENIZER != "aekmeans":
                    raise ValueError(f"{mode} needs VQ_TOKENIZER=aekmeans")
                target = teacher if mode == "dist_ae" else mu[z_train]
                tr, ev = train_student(mode, xt, target, yt, tr_idx, val_idx, xe, device)
            else:
                raise ValueError(f"unknown VQ_MODE {mode}")
            write_h5(path, tr, ev, n_heldin)


if __name__ == "__main__":
    main()
