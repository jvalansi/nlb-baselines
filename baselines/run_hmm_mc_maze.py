"""Poisson HMM for MC_Maze val - the classical discrete-state baseline for the VQ token runs.

Same question as run_vq_ar_transformer_mc_maze.py (what does forcing each 5 ms population
state through one of K prototypes cost?), answered with the textbook model instead of a
tokenizer + transformer: K hidden states, a K x K Markov transition matrix, per-state Poisson
rates over all 182 neurons. Same data, same val split, same K grid.

Fit: Baum-Welch (scaled forward-backward) on train trials with heldin + heldout counts
observed. States k-means-initialized on causal-EMA heldin features (as the VQ k-means
tokenizer). Checkpoint: EM iteration with the best val heldout Poisson NLL of the filtered
rates.

Inference sees heldin only (heldout emissions marginalized out), rates = posterior @ lambda:
  filt    p(z_t | heldin <= t)    causal, same information as the VQ `lookup` / tokenizer
  pred    p(z_t | heldin <= t-1)  causal one-step forecast, same information as the AR models
  smooth  p(z_t | heldin, all t)  acausal, comparable to the masked transformer / NLB entries
"""
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from run_vq_ar_transformer_mc_maze import causal_ema, kmeans, assign, prototype_rates

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
PHASE = "val"
VAL_SPLIT = 0.1
SEED = 0  # val split, identical to the AR / VQ runs
KS = [int(k) for k in os.getenv("HMM_K", "64,256,1024").split(",")]
EM_ITERS = int(os.getenv("HMM_ITERS", "100"))
TAU_MS = 25.0
ALPHA = 10.0  # prototype-rate shrinkage for the init, as in the VQ runs
TRANS_PSEUDO = 1e-3  # Dirichlet pseudo-count on transitions
RATE_FLOOR = 1e-4
SMOKE = int(os.getenv("HMM_SMOKE", "0"))  # >0: use only this many train/eval trials (CPU check)

DIR = Path(__file__).resolve().parents[1] / "outputs"
DIR.mkdir(exist_ok=True)
NPZPATH = DIR.parent / "data" / f"{DATASET_NAME}_{BIN_SIZE_MS}ms.npz"


def emission_loglik(y, lam):
    """y (B, T, n) counts, lam (K, n) rates -> (B, T, K) log p(y_t | z_t=k), up to a k-independent constant."""
    return y @ lam.clamp(min=RATE_FLOOR).log().T - lam.sum(1)


def forward_backward(logB, pi, A, need_beta=True):
    """Scaled forward-backward. logB (B, T, K). Returns alpha (filtered posteriors), beta (scaled),
    log c (B, T), and maxes (B, T) so that log p(y) = sum(log c) + sum(maxes)."""
    Bn, T, K = logB.shape
    m = logB.max(-1, keepdim=True).values
    Bp = (logB - m).exp()
    alpha = torch.empty_like(Bp)
    logc = torch.empty(Bn, T, device=Bp.device)
    a = pi.unsqueeze(0) * Bp[:, 0]
    for t in range(T):
        if t > 0:
            a = (alpha[:, t - 1] @ A) * Bp[:, t]
        c = a.sum(-1, keepdim=True)
        alpha[:, t] = a / c
        logc[:, t] = c.squeeze(-1).log()
    beta = None
    if need_beta:
        beta = torch.empty_like(Bp)
        beta[:, -1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[:, t] = ((Bp[:, t + 1] * beta[:, t + 1]) @ A.T) / logc[:, t + 1].exp().unsqueeze(-1)
    return alpha, beta, logc, m.squeeze(-1), Bp


def em_step(y, pi, A, lam, chunk=256):
    """One Baum-Welch iteration on y (N, T, n). Returns new (pi, A, lam) and train log-lik per bin."""
    K = len(pi)
    g0 = torch.zeros(K, device=y.device)
    xi = torch.zeros(K, K, device=y.device)
    gy = torch.zeros_like(lam)
    gs = torch.zeros(K, device=y.device)
    ll = 0.0
    for i in range(0, len(y), chunk):
        yc = y[i:i + chunk]
        alpha, beta, logc, m, Bp = forward_backward(emission_loglik(yc, lam), pi, A)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(-1, keepdim=True)
        g0 += gamma[:, 0].sum(0)
        for t in range(1, yc.shape[1]):
            w = Bp[:, t] * beta[:, t] / logc[:, t].exp().unsqueeze(-1)
            xi += (alpha[:, t - 1].T @ w) * A
        gy += torch.einsum("btk,btn->kn", gamma, yc)
        gs += gamma.sum((0, 1))
        ll += (logc.sum() + m.sum()).item()
    pi = (g0 + TRANS_PSEUDO) / (g0 + TRANS_PSEUDO).sum()
    A = (xi + TRANS_PSEUDO) / (xi + TRANS_PSEUDO).sum(1, keepdim=True)
    lam = (gy / gs.clamp(min=1e-8).unsqueeze(1)).clamp(min=RATE_FLOOR)
    return pi, A, lam, ll / y.shape[0] / y.shape[1]


@torch.no_grad()
def infer_rates(x, pi, A, lam, n_heldin, chunk=256):
    """Heldin-only posteriors -> rates for all neurons. Returns dict filt/pred/smooth of (N, T, n_all) numpy."""
    out = {"filt": [], "pred": [], "smooth": []}
    for i in range(0, len(x), chunk):
        alpha, beta, _, _, _ = forward_backward(emission_loglik(x[i:i + chunk], lam[:, :n_heldin]), pi, A)
        prior = torch.cat([pi.expand(len(alpha), 1, -1), alpha[:, :-1] @ A], 1)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(-1, keepdim=True)
        for name, post in (("filt", alpha), ("pred", prior), ("smooth", gamma)):
            out[name].append((post @ lam).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def heldout_nll(rates, y, n_heldin):
    r = rates[:, :, n_heldin:]
    return (r - y[:, :, n_heldin:] * r.clamp(min=1e-8).log()).mean().item()


def write_h5(path, train_rates, eval_rates, n_heldin):
    with h5py.File(path, "w") as f:
        g = f.create_group(DATASET_NAME)
        g.create_dataset("train_rates_heldin", data=train_rates[:, :, :n_heldin])
        g.create_dataset("train_rates_heldout", data=train_rates[:, :, n_heldin:])
        g.create_dataset("eval_rates_heldin", data=eval_rates[:, :, :n_heldin])
        g.create_dataset("eval_rates_heldout", data=eval_rates[:, :, n_heldin:])
    print(f"Wrote {path}", flush=True)


@torch.no_grad()
def main():
    np.random.seed(SEED)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz = np.load(NPZPATH)
    x_train = npz["train_heldin"].astype(np.float32)
    y_train = np.concatenate([x_train, npz["train_heldout"].astype(np.float32)], axis=2)
    x_eval = npz["eval_heldin"].astype(np.float32)
    n_train, T, n_heldin = x_train.shape
    perm = np.random.permutation(n_train)  # same split as the AR / VQ runs (same seed, same call order)
    n_val = int(round(n_train * VAL_SPLIT))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if SMOKE:
        tr_idx, val_idx, x_eval = tr_idx[:SMOKE], val_idx[:SMOKE], x_eval[:SMOKE]
    print(f"device {device}  K={KS}  iters={EM_ITERS}  train {len(tr_idx)} val {len(val_idx)} eval {len(x_eval)}", flush=True)

    f = causal_ema(x_train, TAU_MS)
    mu_f, sd_f = f.reshape(-1, n_heldin).mean(0), f.reshape(-1, n_heldin).std(0) + 1e-6
    f = torch.from_numpy((f - mu_f) / sd_f).to(device)
    yt = torch.from_numpy(y_train).to(device)
    xt = torch.from_numpy(x_train).to(device)
    xe = torch.from_numpy(x_eval).to(device)
    y_tr, y_val = yt[tr_idx], yt[val_idx]

    for K in KS:
        feats = f[tr_idx].reshape(-1, n_heldin)
        z = assign(feats, kmeans(feats, K, 50, device))
        lam = prototype_rates(z, y_tr.reshape(-1, y_tr.shape[2]), K, ALPHA)
        zz = z.reshape(len(tr_idx), T)
        A = torch.full((K, K), TRANS_PSEUDO, device=device).index_put_(
            (zz[:, :-1].reshape(-1), zz[:, 1:].reshape(-1)), torch.ones(zz[:, 1:].numel(), device=device), accumulate=True)
        A = A / A.sum(1, keepdim=True)
        pi = torch.bincount(zz[:, 0], minlength=K).float().add(TRANS_PSEUDO)
        pi = pi / pi.sum()

        best, best_params, lls = float("inf"), None, []
        for it in range(EM_ITERS + 1):
            val = heldout_nll(torch.from_numpy(infer_rates(xt[val_idx], pi, A, lam, n_heldin)["filt"]), y_val.cpu(), n_heldin)
            if val < best:
                best, best_params = val, (pi.clone(), A.clone(), lam.clone())
            if it == EM_ITERS:
                break
            pi, A, lam, ll = em_step(y_tr, pi, A, lam)
            lls.append(ll)
            if it % 5 == 0 or it == EM_ITERS - 1:
                print(f"[hmm K={K}] iter {it:3d}  train ll/bin {ll:.4f}  val heldout nll (filt) {val:.5f}  best {best:.5f}", flush=True)
        pi, A, lam = best_params
        used = int((torch.bincount(z, minlength=K) > 0).sum())
        print(f"[hmm K={K}] best val heldout nll {best:.5f}; init states used {used}/{K}", flush=True)
        tr = infer_rates(xt, pi, A, lam, n_heldin)
        ev = infer_rates(xe, pi, A, lam, n_heldin)
        if SMOKE:
            assert all(b >= a - 1e-3 for a, b in zip(lls, lls[1:])), f"EM log-lik decreased: {lls}"
            for name in tr:
                assert np.isfinite(tr[name]).all() and (tr[name] > 0).all(), name
                assert tr[name].shape == (n_train, T, y_train.shape[2]), tr[name].shape
            print("smoke ok", flush=True)
            continue
        for name in ("filt", "pred", "smooth"):
            write_h5(DIR / f"{DATASET_NAME}_hmm_{name}_k{K}_output_{PHASE}.h5", tr[name], ev[name], n_heldin)


if __name__ == "__main__":
    main()
