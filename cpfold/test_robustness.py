#!/usr/bin/env python3
"""
cpfold robustness: missing-allele (9) exactness + wall-clock note.

(1) MISSING-9: inject 9s into donors + recipient, verify the O(N*U) fold still
    reproduces the dense chunkcount (9 is a distinct allele in the substring key;
    recipient-9 -> emission 1.0). The dense matches ChromoPainter's 9-handling
    (Sampler.c L97/L195/L236), so fold-vs-dense exact => fold handles 9 exactly.

(2) WALL-CLOCK CAVEAT: the ~11-16x is an OP-COUNT (algorithmic) speedup. In this
    numpy prototype the fold is SLOWER in wall-clock, because numpy vectorizes the
    dense over all K cheaply while the fold's per-block/per-group Python loops +
    np.add.at carry large constant overhead. A C port realizes the asymptotic win;
    interpreted numpy does not.
"""
import numpy as np, time
from oracle_dense import load_phase, load_recom, parse_ids
from engine_fold import (emissions, dense_forward, dense_backward,
                         chunkcount_per_donor, perpop, fold_ONU, make_blocks)


def main():
    arr, nhap, nsnp, pos = load_phase('win/win.phase')
    lam = load_recom('win/win.recom', nsnp)
    dr, dp, recips, popnames = parse_ids('win/cp.idfile', 'win/cp.poplist')
    K = len(dr); npop = len(popnames); cprob = 1.0 / K; N = nsnp
    T = 1 - np.exp(-(pos[1:] - pos[:-1]) * 400000.0 * lam[:-1])
    rng = np.random.RandomState(0)

    donors9 = arr[dr].copy()
    donors9[rng.random(donors9.shape) < 0.02] = 9
    print(f"MISSING-9: N={N} K={K}, ~2% 9s in donors")
    for B in (50, 100):
        blocks = make_blocks(N, B)
        ccd = np.zeros(npop); cco = np.zeros(npop)
        for h in (recips[0][1], recips[0][2]):
            nh = arr[h].copy(); nh[rng.random(N) < 0.02] = 9
            Eh = emissions(nh, donors9, 0.0006338578)
            ad, Asd = dense_forward(Eh, T, cprob); cd, Bsd = dense_backward(Eh, T, cprob)
            ccd += perpop(chunkcount_per_donor(ad, Asd, cd, Bsd, Eh, T, cprob), dp, npop)
            cc, _, _ = fold_ONU(Eh, T, cprob, donors9, dp, npop, blocks); cco += cc
        rel = (np.abs(cco - ccd) / (np.abs(ccd) + 1e-300)).max()
        print(f"  block={B}: fold-vs-dense WITH 9s max rel err = {rel:.3e}")

    donors = arr[dr]; Eh = emissions(arr[recips[0][1]], donors, 0.0006338578)
    blocks = make_blocks(N, 50)
    t = time.time()
    for _ in range(3):
        ad, Asd = dense_forward(Eh, T, cprob); cd, Bsd = dense_backward(Eh, T, cprob)
        chunkcount_per_donor(ad, Asd, cd, Bsd, Eh, T, cprob)
    td = (time.time() - t) / 3
    t = time.time()
    for _ in range(3):
        fold_ONU(Eh, T, cprob, donors, dp, npop, blocks)
    tf = (time.time() - t) / 3
    print(f"WALL-CLOCK (numpy, N={N} K={K}, block=50): dense={td*1000:.0f}ms fold={tf*1000:.0f}ms "
          f"-> {td/tf:.2f}x (op-count is ~11-16x; numpy overhead caps wall-clock, C port realizes it)")


if __name__ == "__main__":
    main()
