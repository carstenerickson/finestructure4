# Parallelizing the ChromoPainter forward–backward via the Li–Stephens transition structure

**Status:** design note / future work. *Not implemented.* Research-grade; would change the
numerics (calibration re-validation required). Written to revisit later — the gating
step (a low-rank block-map) is not yet worked out here.

---

## 1. Why this exists

`fs cp` (the calibration painter) is **fork-join-bound**, not memory- or compute-bound.
The OpenMP parallelism is a `#pragma omp parallel for` over the ~1004 donor states
(`cp/ChromoPainterSampler.c`, forward ≈ line 94 / backward ≈ line 233), and that pragma sits
**inside the sequential per-SNP loop**. So a thread team forks/joins — and hits a barrier —
**once per SNP** (~3M times/pass on chr11), with only a few microseconds of work between
barriers. With a **single recipient** there is no coarser parallel axis.

Measured consequences (EPYC Genoa, 4 OMP threads):
- per-process scaling peaks at ~4 threads and regresses past it (8t slower than 4t);
- perf @8t: ~50% in libgomp barriers, ~11% real compute;
- on the 16-core/128 GB corpus box the ~30% fork-join overhead is really the *symptom* of a
  core/RAM mismatch (16 cores want 16-way parallelism; 128 GB holds only ~4 full-Alphamat
  chromosomes), so you're forced to bridge with threads (fork-join cost) or more processes
  (needs less RAM/chrom).

The only way to truly remove the per-SNP barrier is to **break the sequential scan** itself.

---

## 2. The transition is diagonal + rank-1

Forward recursion in linear space (the code keeps it log-space with a per-locus max-shift
`large_num` for stability):

```
alpha[t][i] = e[t][i] * ( (1 - c[t]) * alpha[t-1][i]  +  c[t] * p[i] * sum_j alpha[t-1][j] )
```

- `e[t][i]` = emission / `ObsStateProb` at locus `t`, donor `i`  (a **diagonal** factor)
- `c[t]`    = `TransProb[t]`, the per-locus "jump" (recombination) probability
- `p[i]`    = `copy_prob[i]`, the donor copying prior — **constant across loci** for a fixed
  recipient (this matters; see §4)

So `alpha[t] = M[t] alpha[t-1]` with

```
M[t] = diag(e[t]) * [ (1 - c[t]) I  +  c[t] * p 1^T ]
```

i.e. **a diagonal emission times (scaled identity + rank-1)**. This is exactly the structure
the sub-quadratic Li–Stephens algorithms (PBWT, fastLS) exploit.

---

## 3. The parallel-scan idea

`alpha[T] = M[T] M[T-1] ... M[1] alpha[0]` is a **prefix product of linear maps**, and matrix
product is associative — so it's a **scan**. A Blelloch-style parallel scan computes all prefix
products in `O(log T)` parallel steps instead of `T` sequential ones, **eliminating the
per-SNP barrier**.

Practical blocked variant:

1. Partition the `T` loci into `P` blocks.
2. **In parallel across blocks:** compose each block's maps into one block-map
   `M_b = product of M[t] over the block`.
3. **Parallel scan over the `P` block-maps** → `alpha` at every block boundary.
4. **In parallel across blocks:** recompute the within-block `alpha` from the block's
   left-boundary `alpha` (an independent short sequential run per block — *no cross-block
   barrier*).

Step 4 is structurally the **same recompute we already built for checkpointing** (PR #4) —
but here the blocks are **independent** once boundaries are known, so it is embarrassingly
parallel across blocks/processes rather than barrier-synchronized per SNP. The backward pass
is analogous (compose/scan from the other end).

---

## 4. The gating question: does composition stay low-rank?

Composing two `(diag + rank-1)` maps is **in general higher rank**, so a naive block-map costs
`O(N^2)` to store/compose (`N = 1004` → ~1M doubles per block) — too expensive.

**Viability hinges on whether the block-map admits a compact `(diagonal + low-rank)`
representation** for this specific *jump-to-a-fixed-distribution* structure. Intuition that it
might: the jump always lands in the same direction `p` (constant across loci), so the
"new mass" injected each step lives in a low-dimensional subspace; the emission diagonals are
the complication. This is what the sub-quadratic LS literature works out — it needs to be
**derived/confirmed for the ChromoPainter form** before any C work.

- If the rank is bounded (ideally `O(1)`, or small and block-independent): the scan is cheap
  and the fork-join bound is gone.
- If it is **not** bounded: fall back to step-4-only parallelism, which still requires a
  sequential boundary pass (so it does **not** remove the barrier — limited win).

A 1-page derivation + a small NumPy numerical rank check on synthetic data settles this
cheaply.

---

## 5. Numerical considerations

- The production code is log-space with a per-locus max-shift. A scan **reorders the
  operations** → different rounding → **not bit-identical**. Needs calibration-band
  re-validation (same gate as the `float`-Alphamat lever): diff `.cp.chunklengths.out` /
  `.cp.chunkcounts.out` vs the deployed binary across all 22 autosomes.
- Block composition must carry per-block log-scale factors to preserve normalization /
  avoid under/overflow.
- The rank-1 jump term keeps the forward distribution well-conditioned, which helps.

---

## 6. Effort / risk and where it pays off

- **Research-grade.** The §4 derivation is the gating step; everything downstream is ordinary
  (but careful) engineering.
- **Upside:** removes the per-SNP barrier → near-linear thread scaling within a process → lifts
  the core-bound ceiling in the throughput model. Combined with the existing levers
  (config retune, int8, libmvec, checkpointing) it would make the painter scale on
  high-core boxes without the fork-join tax.
- **Downside / cost:** numeric change (re-validation), and if the low-rank property fails the
  win collapses to the modest step-4 form.

---

## 7. Staged plan to revisit

1. **Derive** the block-map representation; confirm a rank bound (paper math + NumPy numerical
   rank check on synthetic 1000-donor data).
2. **Prototype** the blocked scan in Python/NumPy on one real chromosome; verify chunklengths
   match the C output within calibration tolerance.
3. If viable, **port** forward (then backward) to C; benchmark thread scaling vs the current
   per-SNP scheme on the profiling box.
4. **Calibration re-validation** across 22 autosomes; then a PR.

---

## 8. References (verify before relying)

- Li & Stephens 2003, *Genetics* — the copying model.
- Durbin 2014, *Bioinformatics* — PBWT (positional Burrows–Wheeler Transform).
- Lunter 2019, *Bioinformatics* — "fastLS", sub-quadratic Li–Stephens haplotype matching.
- Blelloch 1990 — prefix sums / parallel scan of linear recurrences.
- Särkkä & García-Fernández 2021 — temporal parallelization of Bayesian smoothers
  (associative-scan formulation of forward–backward), as a template for the scan operator.

---

*Companion context:* see the profiling write-up and the open PRs (config retune in
ancestry-pipeline; libmvec, deterministic-sum, and Alphamat-checkpointing in finestructure4)
for the levers that work *without* changing the algorithm. This note is the only path that
removes the fork-join bound itself.
