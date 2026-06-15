#!/usr/bin/env python3
"""
cpfold FOLDED forward (Step 3, core exactness). Proves the Minimac3 block fold of
ChromoPainter's rescaled forward is EXACT vs the dense O(N*K) recursion.

Rescaled forward (matches oracle_dense / Sampler.c, stable O(1) per locus):
  a_0(i)   = copyprob * e_0(i)
  a_l(i)   = e_l(i)*copyprob + e_l(i)*(1-T_{l-1})*r_{l-1}*a_{l-1}(i)
  As_l     = As_{l-1} + log( sum_i a_l(i)*T_l )       [bare sum at last locus]
  r_{l-1}  = exp(As_{l-2} - As_{l-1})                 [r_0 = exp(-As_0)]

KEY: e_l(i), copyprob group-constant within a block; (1-T)*r per-column scalar.
So a_l(i) = G_l(g) + P_l(g)*aentry(i) with group-constant G,P (affine in the
block-entry value aentry(i)=a_{blockstart-1}(i)). The normalizer folds:
  sum_i a_l(i)*T_l = T_l * sum_g ( size_g*G_l(g) + P_l(g)*Sentry_g )   [O(U)]
Per-donor values are materialized ONLY at block boundaries (O(K)/boundary), the
mandatory dense step (soft emission has no hard reset; a summary refold leaks).
"""
import numpy as np
from oracle_dense import load_phase, load_recom, parse_ids, emission


def emissions(newh, donors, mut):
    N = len(newh); K = donors.shape[0]
    E = np.empty((N, K))
    for l in range(N):
        E[l] = emission(newh[l], donors[:, l], mut)
    return E


def dense_forward(E, T, copyprob):
    """Returns a[N,K] (rescaled forward Anew) and As[N] (cumulative loglik)."""
    N, K = E.shape
    a = np.empty((N, K)); As = np.empty(N)
    a[0] = copyprob * E[0]
    As[0] = np.log(np.sum(a[0] * T[0]))
    for l in range(1, N):
        rlm1 = np.exp(As[l - 2] - As[l - 1]) if l >= 2 else np.exp(-As[0])
        a[l] = E[l] * copyprob + E[l] * (1 - T[l - 1]) * rlm1 * a[l - 1]
        s = np.sum(a[l] * T[l]) if l < N - 1 else np.sum(a[l])
        As[l] = As[l - 1] + np.log(s)
    return a, As


def make_blocks(N, B):
    return [(s, min(s + B, N)) for s in range(0, N, B)]


def fold_forward(E, T, copyprob, donors, blocks):
    """Folded forward. Returns reconstructed a[N,K] and As[N] (O(N*U+(N/B)*K))."""
    N, K = E.shape
    a_rec = np.empty((N, K))     # reconstructed per-donor (for verification)
    As = np.empty(N)
    aprev = None                 # a_{blockstart-1}(i) per donor (materialized)
    for (sblk, eblk) in blocks:
        # group donors by allele substring over this block
        sub = donors[:, sblk:eblk]
        _, gid = np.unique(sub, axis=0, return_inverse=True)
        gid = gid.ravel()
        U = gid.max() + 1
        size_g = np.bincount(gid, minlength=U).astype(float)
        # representative donor per group (emission is group-constant)
        rep = np.zeros(U, dtype=int)
        rep[gid] = np.arange(K)              # last-wins; any rep is fine
        # Sentry_g = sum_{i in g} aprev(i)
        if aprev is None:
            Sentry = np.zeros(U)
        else:
            Sentry = np.bincount(gid, weights=aprev, minlength=U)
        G = np.zeros(U); P = np.zeros(U)
        for l in range(sblk, eblk):
            eg = E[l][rep]                    # e_l(g)
            if l == sblk:
                if aprev is None:             # very first locus, l==0
                    G = copyprob * eg; P = np.zeros(U)
                else:
                    rlm1 = np.exp(As[l - 2] - As[l - 1]) if l >= 2 else np.exp(-As[0])
                    G = eg * copyprob
                    P = eg * (1 - T[l - 1]) * rlm1
            else:
                rlm1 = np.exp(As[l - 2] - As[l - 1]) if l >= 2 else np.exp(-As[0])
                fac = eg * (1 - T[l - 1]) * rlm1
                G = eg * copyprob + fac * G
                P = fac * P
            # folded normalizer -> As[l]
            sumA = np.sum(size_g * G + P * Sentry)
            s = sumA * T[l] if l < N - 1 else sumA
            As[l] = (As[l - 1] if l >= 1 else 0.0) + np.log(s)
            # reconstruct per-donor a_l(i) for verification (interior uses affine form)
            a_rec[l] = G[gid] + (P[gid] * aprev if aprev is not None else 0.0)
        # materialize block-end per-donor values -> entry for next block (O(K))
        aprev = G[gid] + (P[gid] * aprev if aprev is not None else 0.0)
    return a_rec, As


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--recom", required=True)
    ap.add_argument("--idfile", required=True)
    ap.add_argument("--poplist", required=True)
    ap.add_argument("--readNe", type=float, default=400000)
    ap.add_argument("--mut", type=float, default=0.0006338578)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--hap", type=int, default=0)
    a = ap.parse_args()

    arr, nhap, nsnp, pos = load_phase(a.phase)
    lam = load_recom(a.recom, nsnp)
    dr, dp, recips, popnames = parse_ids(a.idfile, a.poplist)
    K = len(dr); donors = arr[dr]
    rhobar = a.readNe; copyprob = 1.0 / K
    d = pos[1:] - pos[:-1]; T = 1 - np.exp(-d * rhobar * lam[:-1])
    newh = arr[a.hap]
    E = emissions(newh, donors, a.mut)

    a_d, As_d = dense_forward(E, T, copyprob)
    blocks = make_blocks(nsnp, a.block)
    a_f, As_f = fold_forward(E, T, copyprob, donors, blocks)

    relA = np.abs(a_f - a_d) / (np.abs(a_d) + 1e-300)
    relS = np.abs(As_f - As_d) / (np.abs(As_d) + 1e-300)
    print(f"FOLD FORWARD vs DENSE  N={nsnp} K={K} block={a.block} nblocks={len(blocks)}")
    print(f"  per-donor a_l(i) max rel err = {relA.max():.3e}")
    print(f"  cumulative As_l  max rel err = {relS.max():.3e}")
    print(f"  final loglik  dense={As_d[-1]:.8f} fold={As_f[-1]:.8f}")
    ok = relA.max() < 1e-9
    print("VERDICT:", "PASS - forward fold is EXACT (affine block decomposition)" if ok else "FAIL")
