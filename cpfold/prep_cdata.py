#!/usr/bin/env python3
"""
Dump the prepared ChromoPainter inputs into a flat binary the standalone C
engine (cpfold.c) reads, so the C is purely the numerical core (no idfile/recom
parsing). Reuses the validated oracle_dense loaders. Also writes the dense
reference per-pop chunkcounts (from the Python dense engine) for a port-defect
check in C.

Layout (little-endian):
  int32  K            # donor haps
  int32  N            # SNPs
  int32  npop
  int32  nrecip       # recipient haps (2 for one diploid individual)
  float64 rhobar, mut, copyprob
  uint8  donors[K*N]
  uint8  recip[nrecip*N]
  float64 pos[N]
  float64 lam[N]       # per-bp recom rate (lam[N-1] unused)
  int32  pop_vec[K]
  float64 ref_cc[npop] # Python dense per-pop chunkcount summed over recip haps
"""
import sys, struct, argparse
import numpy as np
from oracle_dense import load_phase, load_recom, parse_ids
from engine_fold import emissions, dense_forward, dense_backward, chunkcount_per_donor, perpop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--recom", required=True)
    ap.add_argument("--idfile", required=True)
    ap.add_argument("--poplist", required=True)
    ap.add_argument("--readNe", type=float, default=400000)
    ap.add_argument("--mut", type=float, default=0.0006338578)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arr, nhap, nsnp, pos = load_phase(a.phase)
    lam = load_recom(a.recom, nsnp)
    dr, dp, recips, popnames = parse_ids(a.idfile, a.poplist)
    K = len(dr); npop = len(popnames); rhobar = a.readNe; copyprob = 1.0 / K
    donors = arr[dr].astype(np.uint8)
    rec_rows = [recips[0][1], recips[0][2]]
    recip = arr[rec_rows].astype(np.uint8)
    pop_vec = dp.astype(np.int32)

    T = 1 - np.exp(-(pos[1:] - pos[:-1]) * rhobar * lam[:-1])
    ref = np.zeros(npop)
    for r in range(recip.shape[0]):
        E = emissions(recip[r], donors, a.mut)
        ad, Asd = dense_forward(E, T, copyprob); cd, Bsd = dense_backward(E, T, copyprob)
        ref += perpop(chunkcount_per_donor(ad, Asd, cd, Bsd, E, T, copyprob), pop_vec, npop)

    with open(a.out, "wb") as f:
        f.write(struct.pack("<4i", K, nsnp, npop, recip.shape[0]))
        f.write(struct.pack("<3d", rhobar, a.mut, copyprob))
        f.write(donors.tobytes())
        f.write(recip.tobytes())
        f.write(pos.astype(np.float64).tobytes())
        f.write(lam.astype(np.float64).tobytes())
        f.write(pop_vec.tobytes())
        f.write(ref.astype(np.float64).tobytes())
    print(f"wrote {a.out}: K={K} N={nsnp} npop={npop} nrecip={recip.shape[0]} "
          f"rhobar={rhobar} mut={a.mut}")
    print("  python dense ref per-pop:", " ".join(f"{v:.6f}" for v in ref))
    print("  pops:", popnames)


if __name__ == "__main__":
    main()
