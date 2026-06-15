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


def dense_backward(E, T, copyprob):
    """Rescaled backward c_l(i) (=1 + (1-T)*e*rb*c_{l+1}) + Bs[l] cumulative.
    c[N-1]=1. Matches oracle_dense backward (Sampler.c L185-295)."""
    N, K = E.shape
    c = np.empty((N, K)); Bs = np.empty(N)
    c[N - 1] = 1.0
    Bs[N - 1] = np.log(np.sum(T[N - 2] * copyprob * E[N - 1]))   # init Betasum
    for l in range(N - 2, -1, -1):
        rb = np.exp(Bs[l + 2] - Bs[l + 1]) if l + 2 <= N - 1 else np.exp(-Bs[N - 1])
        c[l] = 1.0 + (1 - T[l]) * E[l + 1] * rb * c[l + 1]
        if l > 0:
            Bs[l] = Bs[l + 1] + np.log(np.sum(T[l - 1] * copyprob * E[l] * c[l]))
        else:
            Bs[0] = Bs[1]   # not used; keep array filled
    return c, Bs


def fold_backward(E, T, copyprob, donors, blocks):
    """Folded backward. c_l(i) within a block is affine in the boundary product
    w(i) = E[eblk](i)*c_{eblk}(i): the top locus uses E[eblk] (next block, NOT
    group-constant), so the per-donor E[eblk] must be carried in w, not folded."""
    N, K = E.shape
    c_rec = np.empty((N, K)); Bs = np.empty(N)
    c_rec[N - 1] = 1.0
    Bs[N - 1] = np.log(np.sum(T[N - 2] * copyprob * E[N - 1]))
    cexit_full = np.ones(K)          # c_{eblk}(i); rightmost seed c[N-1]=1
    for (sblk, eblk) in reversed(blocks):
        sub = donors[:, sblk:eblk]
        _, gid = np.unique(sub, axis=0, return_inverse=True)
        gid = gid.ravel(); U = gid.max() + 1
        size_g = np.bincount(gid, minlength=U).astype(float)
        rep = np.zeros(U, dtype=int); rep[gid] = np.arange(K)
        top = min(eblk - 1, N - 2)                   # l=N-1 is the seed c=1
        # boundary product carried per-donor: w(i) = E[top+1](i) * c_{top+1}(i)
        w_full = E[top + 1] * cexit_full
        Sw = np.bincount(gid, weights=w_full, minlength=U)
        GB = np.zeros(U); PB = np.zeros(U)
        for l in range(top, sblk - 1, -1):
            rb = np.exp(Bs[l + 2] - Bs[l + 1]) if l + 2 <= N - 1 else np.exp(-Bs[N - 1])
            if l == top:
                GB = np.ones(U)
                PB = np.full(U, (1 - T[l]) * rb)     # scalar: E[top+1] absorbed into w
            else:
                fac = (1 - T[l]) * E[l + 1][rep] * rb   # E[l+1] in-block, group-constant
                GB = 1.0 + fac * GB
                PB = fac * PB
            if l > 0:
                egl = E[l][rep]
                sB = np.sum(egl * (size_g * GB + PB * Sw))
                Bs[l] = Bs[l + 1] + np.log(T[l - 1] * copyprob * sB)
            c_rec[l] = GB[gid] + PB[gid] * w_full
        cexit_full = c_rec[sblk]                     # c_{sblk} = exit for next (leftward) block
    return c_rec, Bs


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


def chunkcount_per_donor(a, As, c, Bs, E, T, copyprob):
    """Per-donor corrected_chunk_count from the rescaled forward a + backward c.
    cc[i] = sum_l (a_{l+1} c_{l+1} KF1_l - a_l c_{l+1} KF_l E[l+1] (1-T[l])) + start.
      KF1_l = exp(As[l]   + BsR(l+2) - As[N-1]),  KF_l = exp(As[l-1] + BsR(l+2) - As[N-1])
      start = a_0 c_0 exp(Bs[1] - As[N-1]);  As[-1]=0, BsR(N)=0, c[N-1]=1."""
    N, K = E.shape
    Asf = As[N - 1]
    cc = np.zeros(K)
    for l in range(N - 1):
        BsR = Bs[l + 2] if l + 2 <= N - 1 else 0.0
        Asm1 = As[l - 1] if l >= 1 else 0.0
        KF1 = np.exp(As[l] + BsR - Asf)
        KF = np.exp(Asm1 + BsR - Asf)
        clp1 = c[l + 1]
        cc += a[l + 1] * clp1 * KF1 - a[l] * clp1 * KF * E[l + 1] * (1 - T[l])
    cc += a[0] * c[0] * np.exp(Bs[1] - Asf)
    return cc


def perpop(cc, dp, npop):
    return np.array([cc[dp == p].sum() for p in range(npop)])


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
    ap.add_argument("--ref", default=None, help="fs cp .chunkcounts.out for binary comparison")
    a = ap.parse_args()

    arr, nhap, nsnp, pos = load_phase(a.phase)
    lam = load_recom(a.recom, nsnp)
    dr, dp, recips, popnames = parse_ids(a.idfile, a.poplist)
    K = len(dr); donors = arr[dr]
    rhobar = a.readNe; copyprob = 1.0 / K
    d = pos[1:] - pos[:-1]; T = 1 - np.exp(-d * rhobar * lam[:-1])

    blocks = make_blocks(nsnp, a.block)
    npop = len(popnames)
    # per-pop chunkcount summed over both recipient haps, dense vs fold
    cc_dense = np.zeros(npop); cc_fold = np.zeros(npop)
    maxA = maxC = 0.0
    for h in (recips[0][1], recips[0][2]):
        Eh = emissions(arr[h], donors, a.mut)
        a_d, As_d = dense_forward(Eh, T, copyprob)
        a_f, As_f = fold_forward(Eh, T, copyprob, donors, blocks)
        c_d, Bs_d = dense_backward(Eh, T, copyprob)
        c_f, Bs_f = fold_backward(Eh, T, copyprob, donors, blocks)
        maxA = max(maxA, (np.abs(a_f - a_d) / (np.abs(a_d) + 1e-300)).max())
        maxC = max(maxC, (np.abs(c_f - c_d) / (np.abs(c_d) + 1e-300)).max())
        cc_dense += perpop(chunkcount_per_donor(a_d, As_d, c_d, Bs_d, Eh, T, copyprob), dp, npop)
        cc_fold += perpop(chunkcount_per_donor(a_f, As_f, c_f, Bs_f, Eh, T, copyprob), dp, npop)

    print(f"FOLD vs DENSE  N={nsnp} K={K} block={a.block} nblocks={len(blocks)} pops={popnames}")
    print(f"  state fold: max rel err  forward a={maxA:.2e}  backward c={maxC:.2e}")
    relCC = (np.abs(cc_fold - cc_dense) / (np.abs(cc_dense) + 1e-300)).max()
    print(f"  chunkcount dense: " + " ".join(f"{v:.6f}" for v in cc_dense))
    print(f"  chunkcount fold : " + " ".join(f"{v:.6f}" for v in cc_fold))
    print(f"  fold-vs-dense chunkcount max rel err = {relCC:.3e}")
    if a.ref:
        with open(a.ref) as f:
            hdr = f.readline().split()[1:]; row = f.readline().split()
            refv = np.array([float(x) for x in row[1:]])
        idx = [popnames.index(p) for p in hdr]
        relbin = (np.abs(cc_fold[idx] - refv) / (np.abs(refv) + 1e-300)).max()
        print(f"  binary ({hdr}): " + " ".join(f"{v:.6f}" for v in refv))
        print(f"  fold-vs-BINARY max rel err = {relbin:.3e}")
    ok = maxA < 1e-9 and maxC < 1e-9 and relCC < 1e-9
    print("VERDICT:", "PASS - fold reproduces dense chunkcounts EXACTLY" if ok else "FAIL")
