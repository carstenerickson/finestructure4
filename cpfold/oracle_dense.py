#!/usr/bin/env python3
"""
cpfold dense ORACLE: a faithful numpy port of ChromoPainter's loglik forward-
backward + chunk-count (cp/ChromoPainterSampler.c), validated against the real
`fs cp` binary output. This is the ground-truth the folded engine must match.

Ported verbatim (DO NOT substitute textbook Li-Stephens forms):
- emission      Sampler.c L97-103 : e = (1-mut)*[match] + mut*[mismatch]; 9->1; 8->SMALL
- transition    deltaLiStephens L33-40 : T[l] = 1-exp(-(pos[l+1]-pos[l])*rhobar*lambda[l])
- forward init  L62-65 : Alpha[0][i]=log(copyprobSTART[i]*e0); Alphasum=log(sum exp(Alpha0)*T0)
- forward       L107-112 : Anew = e*copyprob + e*(1-T[l-1])*exp(Alpha[l-1]+large); rescale
- backward init L195-204 : BetaPREV=0; Betasum=log(sum T[N-2]*copyprob*e_{N-1})
- backward      L252 : BetaCUR=log(exp(Betasum+large)+(1-T)*e_{l+1}*exp(BetaPREV+large))-large
- chunkcount    L273 : cc += exp(a_{l+1}+b_{l+1}-As) - exp(a_l+b_{l+1}-As)*e_{l+1}*(1-T[l])
- start term    L259,L348 : cc += exp(a_0+b_0-As)
- params        rhobar=readN_e/nhaps (L721), copyprob=copyprobSTART=1/ndonors, mut=-M

Run config (deterministic): fs cp ... -s 0 -i 0 -n <readNe> -M <mut>.
"""
import sys, argparse
import numpy as np

SMALL_NUM = 1e-20


def load_phase(path):
    with open(path, "rb") as f:
        nhap = int(f.readline()); nsnp = int(f.readline())
        pline = f.readline().split()
        pos = np.array([float(x) for x in pline[1:nsnp + 1]])
        arr = np.empty((nhap, nsnp), dtype=np.uint8)
        for i in range(nhap):
            row = f.readline().strip()
            arr[i] = np.frombuffer(row[:nsnp], dtype=np.uint8) - ord('0')
    return arr, nhap, nsnp, pos


def load_recom(path, nsnp):
    """start.pos recom.rate.perbp ; lambda[l] = rate between snp l and l+1."""
    lam = np.zeros(nsnp)
    with open(path) as f:
        f.readline()  # header
        for l, line in enumerate(f):
            if l >= nsnp:
                break
            p = line.split()
            lam[l] = float(p[1])
    return lam


def parse_ids(idfile, poplist):
    """Return donor row indices, donor pop indices, recipient row pairs, pop names."""
    popidx = {}
    with open(poplist) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[1] == "D":
                if p[0] not in popidx:
                    popidx[p[0]] = len(popidx)
    popnames = sorted(popidx, key=popidx.get)
    donors, donor_pop, recips = [], [], []
    with open(idfile) as f:
        for ind, line in enumerate(f):
            p = line.split()
            name, pop, inc = p[0], p[1], int(p[2])
            r0, r1 = 2 * ind, 2 * ind + 1
            if pop == "TARGET":
                recips.append((name, r0, r1))
            elif inc == 1 and pop in popidx:
                donors += [r0, r1]
                donor_pop += [popidx[pop], popidx[pop]]
    return np.array(donors), np.array(donor_pop), recips, popnames


def emission(newh_l, donors_l, mut):
    """e(i) for one locus. newh_l scalar, donors_l vec. 0/1 alleles + 9/8 specials."""
    if newh_l == 9:
        return np.ones(donors_l.shape[0])
    match = (donors_l == newh_l)
    if newh_l == 8:
        return (1 - SMALL_NUM) * match + SMALL_NUM * (~match)
    return (1 - mut) * match + mut * (~match)


def paint_hap(newh, donors, pos, lam, rhobar, copyprob, mut):
    """Forward + backward + chunkcount for one recipient hap. Returns cc[K]."""
    K, N = donors.shape
    # transition T[l] over interval [l,l+1], l=0..N-2
    d = pos[1:] - pos[:-1]
    T = 1.0 - np.exp(-d * rhobar * lam[:-1])   # len N-1
    # emissions per locus cached as (N,K) is too big? N*K=2000*1004=2M ok
    E = np.empty((N, K))
    for l in range(N):
        E[l] = emission(newh[l], donors[:, l], mut)

    # ---- forward (Alpha stored as log, rescaled) ----
    Alpha = np.empty((N, K))
    Alpha[0] = np.log(copyprob * E[0])
    Alphasum = np.log(np.sum(np.exp(Alpha[0]) * T[0]))
    for l in range(1, N):
        large = -Alphasum
        Anew = E[l] * copyprob + E[l] * (1 - T[l - 1]) * np.exp(Alpha[l - 1] + large)
        Alpha[l] = np.log(Anew) - large
        if l < N - 1:
            Alphasumnew = np.sum(Anew * T[l])
        else:
            Alphasumnew = np.sum(Anew)
        Alphasum = np.log(Alphasumnew) - large
    As = Alphasum  # final total log-normalizer (constant for backward)

    # ---- backward + chunkcount ----
    cc = np.zeros(K)
    BetaPREV = np.zeros(K)                       # log beta at locus N-1
    Betasum = np.log(np.sum(T[N - 2] * copyprob * E[N - 1] * np.exp(BetaPREV)))
    cpnewSTART = None
    for l in range(N - 2, -1, -1):
        large = -Betasum
        ePREV = E[l + 1]
        BetaCUR = np.log(np.exp(Betasum + large) +
                         (1 - T[l]) * ePREV * np.exp(BetaPREV + large)) - large
        e_a_lp1_bp = np.exp(Alpha[l + 1] + BetaPREV - As)
        e_a_l_bp = np.exp(Alpha[l] + BetaPREV - As)
        cc += e_a_lp1_bp - e_a_l_bp * ePREV * (1 - T[l])
        if l == 0:
            cpnewSTART = np.exp(Alpha[0] + BetaCUR - As)
        if l > 0:
            Betasum = np.log(np.sum(T[l - 1] * copyprob * E[l] * np.exp(BetaCUR + large))) - large
        BetaPREV = BetaCUR
    cc += cpnewSTART
    return cc


def run(phase, recom, idfile, poplist, readNe, mut, rhobar_override=None):
    arr, nhap, nsnp, pos = load_phase(phase)
    lam = load_recom(recom, nsnp)
    donor_rows, donor_pop, recips, popnames = parse_ids(idfile, poplist)
    npop = len(popnames)
    K = len(donor_rows)
    nhaps = K               # donor haps for this recipient (L635/L721)
    # Empirically (oracle vs binary), with -i 0 -n X the engine uses rhobar = X
    # DIRECTLY (ne_find=1 path, L722 N_e=Par->N_e=readN_e), NOT readN_e/nhaps.
    # The "N_e=398.4" in the log is the post-division DEFAULT report, a red herring.
    rhobar = rhobar_override if rhobar_override is not None else readNe
    copyprob = 1.0 / K
    donors = arr[donor_rows]
    results = {}
    for name, r0, r1 in recips:
        cc_tot = np.zeros(K)
        for r in (r0, r1):
            cc_tot += paint_hap(arr[r], donors, pos, lam, rhobar, copyprob, mut)
        out = np.zeros(npop)
        for p in range(npop):
            out[p] = cc_tot[donor_pop == p].sum()
        results[name] = out
    return results, popnames, K, rhobar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--recom", required=True)
    ap.add_argument("--idfile", required=True)
    ap.add_argument("--poplist", required=True)
    ap.add_argument("--readNe", type=float, required=True, help="the -n value passed to fs cp")
    ap.add_argument("--mut", type=float, required=True, help="the -M value")
    ap.add_argument("--rhobar", type=float, default=None, help="override rhobar directly")
    ap.add_argument("--ref", default=None, help="fs cp .chunkcounts.out to compare")
    a = ap.parse_args()

    results, popnames, K, rhobar = run(a.phase, a.recom, a.idfile, a.poplist, a.readNe, a.mut, a.rhobar)
    print(f"oracle: K={K} donors, rhobar={rhobar:.6f}, pops={popnames}")
    for name, out in results.items():
        print(f"  {name}: " + " ".join(f"{v:.6f}" for v in out))

    if a.ref:
        with open(a.ref) as f:
            header = f.readline().split()[1:]
            refrow = f.readline().split()
            refname = refrow[0]
            refvals = np.array([float(x) for x in refrow[1:]])
        # align oracle pop order to ref header order
        oracle = results[refname]
        idx = [popnames.index(p) for p in header]
        oracle_aligned = oracle[idx]
        rel = np.abs(oracle_aligned - refvals) / (np.abs(refvals) + 1e-300)
        print(f"\nref header: {header}")
        print(f"ref  : " + " ".join(f"{v:.6f}" for v in refvals))
        print(f"oracl: " + " ".join(f"{v:.6f}" for v in oracle_aligned))
        print(f"max rel err vs binary = {rel.max():.3e}   (gate < 5e-6)")
        print("VERDICT:", "PASS - oracle faithful" if rel.max() < 5e-6 else "FAIL - port bug (check rhobar units, emission, indexing)")
