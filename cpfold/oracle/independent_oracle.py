#!/usr/bin/env python3
"""
INDEPENDENT ORACLE for ChromoPainter corrected chunk counts.

Purpose: validate `fs cp` (dense AND -fold) against a reference that shares NO code
with finestructure4 - it implements the Li-Stephens copying HMM from the model
definition and computes the per-pop chunk counts by EXACT brute-force enumeration
over all K^N donor paths. This catches a shared bug in the emission / transition /
chunk-count code (which a fold-vs-dense comparison cannot, since both share cp_emis).

Tiny case (cpfold/oracle/data.*): 1 recipient (TGT) vs 4 donor haplotypes in 2 pops,
6 loci, including a missing allele (9) in both a recipient locus and a donor locus to
exercise the r==9 emission branch with independent code.

Cross-checked by an independent Monte-Carlo implementation (formula-free: it samples
copying paths and counts recombination events directly); both agree with fs:
  brute-force  popA=1.4288035005  popB=1.2117673272   (exact)
  monte-carlo  popA=1.428518      popB=1.212272        (4e6 samples, std err <1e-3)
  fs dense/fold popA=1.428804     popB=1.211767        (printed, 6 dp)
"""
import numpy as np
from itertools import product

# ---- DATA (cpfold/oracle/data.phase) ----
TGT = [0, 0, 9, 1, 1, 0]          # recipient; 9 = missing
donors = {
    'A1': [0, 0, 0, 1, 0, 0],     # popA
    'A2': [0, 1, 0, 1, 0, 1],     # popA
    'B1': [1, 1, 1, 0, 1, 0],     # popB
    'B2': [1, 0, 1, 9, 1, 0],     # popB; 9 = missing donor allele
}
donor_names = ['A1', 'A2', 'B1', 'B2']
donor_mat = np.array([donors[n] for n in donor_names])   # (K=4, N=6)
pos = [100, 200, 400, 700, 1100, 1600]
lam = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]

# ---- PARAMETERS (fs cp ... -j -i 0 -n 100 -M 0.01) ----
K, N = 4, 6
rho = 100.0
mut = 0.01
copy_prob = np.full(K, 1.0 / K)        # jump-target prob = 1/K
copy_probSTART = np.full(K, 1.0 / K)   # locus-0 prior   = 1/K
popA_idx, popB_idx = [0, 1], [2, 3]

# ---- EMISSION e(r,d) ----
def emit(r, d):
    if r == 9:    return 1.0           # recipient missing => uninformative
    if r == d:    return 1.0 - mut
    return mut

E = np.array([[emit(TGT[l], donor_mat[i][l]) for i in range(K)] for l in range(N)])

# ---- TRANSITION: T[l] over interval (l, l+1] ----
T = np.array([1.0 - np.exp(-(pos[l + 1] - pos[l]) * rho * lam[l]) for l in range(N - 1)])

def trans_prob(j, i, l):   # P(i at l+1 | j at l)
    return (1.0 - T[l]) * (1.0 if i == j else 0.0) + T[l] * copy_prob[i]

# ---- EXACT BRUTE FORCE over all K^N paths ----
Z = 0.0
chunk_accum = np.zeros(K)
for path in product(range(K), repeat=N):
    w = copy_probSTART[path[0]] * E[0][path[0]]
    for l in range(1, N):
        w *= trans_prob(path[l - 1], path[l], l - 1) * E[l][path[l]]
    if w == 0.0:
        continue
    Z += w
    starts = np.zeros(K)
    starts[path[0]] += 1.0                         # locus 0 is always a chunk start
    for l in range(1, N):
        j, i = path[l - 1], path[l]
        if i != j:
            starts[i] += 1.0                       # donor change => new chunk
        else:                                      # same donor: split no-jump vs jump-repick
            jump = T[l - 1] * copy_prob[i]
            starts[i] += jump / ((1.0 - T[l - 1]) + jump)
    chunk_accum += w * starts

chunkcount = chunk_accum / Z
popA = sum(chunkcount[i] for i in popA_idx)
popB = sum(chunkcount[i] for i in popB_idx)

if __name__ == '__main__':
    print(f"forward_loglik = {np.log(Z):.15f}")
    for i, n in enumerate(donor_names):
        print(f"  {n}: {chunkcount[i]:.10f}")
    print(f"popA (A1+A2) = {popA:.10f}")
    print(f"popB (B1+B2) = {popB:.10f}")
