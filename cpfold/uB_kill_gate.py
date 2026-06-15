#!/usr/bin/env python3
"""
cpfold ECONOMICS KILL GATE (cheapest-decisive-first).

ChromoPainter runs a DENSE Li-Stephens forward-backward over ALL K donors x N
SNPs - that O(N*K) FB is its bottleneck (unlike SparsePainter, painting IS the
cost). The Minimac3 unique-haplotype block fold collapses the K donors to U
distinct local allele-substrings per block. This script measures U(B) on the
REAL donor panel and computes the BOUNDARY-OVERHEAD-ADJUSTED net speedup,
BEFORE writing any folded engine.

Cost model (build-amortized: the panel fold maps are target-independent, built
once and reused for every recipient, like 3b-A):
  dense per recipient        ~ N*K
  folded per recipient       ~ N*U  (FB on U states) + (N/B)*Cunfold  (boundary)
where Cunfold is the per-boundary unfold/refold cost. We bracket Cunfold in
units of K (c*K): c=1 (cheap, O(K) redistribute), c=2 (conservative).
  net speedup(B) = K / ( mean_U(B) + c*K/B )

KILL if the best net speedup over B is not comfortably > 1 (and ideally beats
kalis's exact-dense constant factor + windowing's ~30x).
"""
import sys, argparse, statistics
import numpy as np


def load_phase(path, maxhaps=None):
    """ChromoPainter/SparsePainter .phase: line1=nhap, line2=nsnp, line3=P...,
    then nhap allele rows of '0'/'1'/'9'. Returns uint8 array (nhap x nsnp)."""
    with open(path, "rb") as f:
        nhap = int(f.readline())
        nsnp = int(f.readline())
        f.readline()  # P-line
        if maxhaps:
            nhap = min(nhap, maxhaps)
        arr = np.empty((nhap, nsnp), dtype=np.uint8)
        for i in range(nhap):
            row = f.readline().strip()
            arr[i] = np.frombuffer(row[:nsnp], dtype=np.uint8)
    return arr, nhap, nsnp


def u_of_block(arr, a, b):
    """Number of distinct donor allele-substrings (rows) over columns [a,b)."""
    return np.unique(arr[:, a:b], axis=0).shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--blocks", default="20,50,100,200,500,1000,2000")
    ap.add_argument("--maxblocks", type=int, default=400,
                    help="cap blocks sampled per size (speed); -1 = all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arr, nhap, nsnp = load_phase(a.phase)
    K = nhap
    L = []
    def log(s=""):
        L.append(s); print(s)

    log(f"cpfold ECONOMICS KILL GATE   panel K={K} haps  N={nsnp} SNPs")
    log("net speedup(B) = K / (mean_U + c*K/B),  c=boundary cost in units of K")
    log("")
    log(f"{'B':>6} {'nblk':>6} {'meanU':>8} {'medU':>6} {'maxU':>6} "
        f"{'K/U':>7} {'net(c=1)':>9} {'net(c=2)':>9}")

    best = (0.0, None)
    rows = []
    for B in [int(x) for x in a.blocks.split(",")]:
        starts = list(range(0, nsnp - B + 1, B))
        if a.maxblocks != -1 and len(starts) > a.maxblocks:
            # evenly subsample block starts for speed
            idx = np.linspace(0, len(starts) - 1, a.maxblocks).astype(int)
            sample = [starts[i] for i in idx]
        else:
            sample = starts
        Us = [u_of_block(arr, s, s + B) for s in sample]
        meanU = statistics.fmean(Us)
        medU = statistics.median(Us)
        maxU = max(Us)
        raw = K / meanU
        net1 = K / (meanU + 1 * K / B)
        net2 = K / (meanU + 2 * K / B)
        log(f"{B:>6} {len(starts):>6} {meanU:>8.1f} {medU:>6.0f} {maxU:>6} "
            f"{raw:>7.1f} {net1:>9.1f} {net2:>9.1f}")
        rows.append((B, meanU, raw, net1, net2))
        if net2 > best[0]:
            best = (net2, B)

    log("")
    log(f"BEST conservative (c=2) net speedup = {best[0]:.1f}x at block B={best[1]} SNPs")
    log("Compare to: kalis exact-dense constant factor (hardware), windowing ~30x approx.")
    verdict = ("PROCEED: net fold materially > 1 and competitive"
               if best[0] >= 5 else
               "WEAK: net fold marginal after boundary overhead - reconsider")
    log(f"VERDICT: {verdict}")

    report = "\n".join(L)
    if a.out:
        with open(a.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
