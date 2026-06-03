# Float Alphamat via linear-space rescaling — design pass & feasibility

**Status:** design / feasibility. *Not implemented.* Verdict: **feasible and attractive**
(combines a ~−14% throughput win with elimination of the ~24% `exp`/`log` hotspot, and is
*more* numerically stable than today). Changes numerics → calibration re-validation required.
The delicate part is the backward chunk-count formulas, not the idea.

---

## 1. Why the naïve `double → float` Alphamat is dead (measured 5.8%)

`forwardAlgorithm` stores, per locus/donor,
```
Alphamat[t][i] = log(Anew[t][i]) - large_num   where large_num = -Alphasum_{t-1}
              = log(alpha[t][i]) + Sigma_{t-1}
```
i.e. **the log of the (small) forward value plus the large cumulative log-normalizer
`Sigma`**. On real chr11 (100k loci × real recom map) `|Sigma|` reaches **~1e6**. The forward
*read* one locus later does
```
prev[i] = exp( Alphamat[t-1][i] + large_num_t )      # = alpha[t-1][i]/Snew_{t-1}, normalized
```
Storing `Alphamat` as **float32** rounds a `~1e6`-magnitude value with absolute error
`|Sigma|·6e-8 ≈ 0.06`; that lands in the *exponent* of the `exp`, so `prev` carries a
**~6% relative error**. Measured / modeled:

| `|offset|` | log-space float32 err | rescale float32 err |
|--:|--:|--:|
| 1e4 | 0.05% | 1e-7 % |
| 1e5 | 0.39% | 1e-7 % |
| 1e6 | 3.2% | 1e-7 % |
| 2e6 | 6.4% | 1e-7 % |

(Measured on the box: drop-in float = **5.84%** chunklength Δ — matches `|Sigma|`≈1.7e6.)

The signal we care about (`log(alpha)`, O(10)) lives in the **low bits below a ~1e6 offset**,
which float32 cannot hold. That is the whole problem.

---

## 2. The fix: store the *normalized* forward in linear space

This is the textbook **scaled forward–backward** (Rabiner 1989, §V.A). Key observation: the
current code is *already implicitly normalized* — `prev[i] = exp(Alphamat[t-1][i]+large_num)`
equals `Anew[t-1][i] / Snew_{t-1}`, an O(1) normalized value. So instead of reconstructing it
from a log+huge-offset representation, **store it directly**:

```
ahat[t][i]  : normalized forward, O(1), stored as FLOAT          (the 12 GB matrix)
logScale[t] : cumulative log-normalizer, DOUBLE scalar/array      (tiny: T doubles)
```

New forward recursion — **no per-element `exp`/`log`**:
```
Anew[i]   = e[t][i] * ( copy_prob[i] + (1 - c[t-1]) * ahat[t-1][i] )   # read ahat directly
Snew      = sum_i Anew[i] * c[t]            # = the existing Alphasumnew
ahat[t][i]= Anew[i] / Snew                  # store (float); replaces log(Anew)-large_num
logScale += log(Snew)                       # one log per locus (was the Alphasum update)
```
`ahat` values are O(1) → float32 holds them at ~1e-7 relative, **independent of `|Sigma|`**
(table above, right column). The large magnitude now lives in `logScale` (double), exactly.

The likelihood the EM uses (`Alphasum`) **is** `logScale` — same interface, so `-in`/`-iM`
estimation is unchanged.

---

## 3. Two wins, not one

1. **Memory:** forward Alphamat `24 → 12 GB` (chr11). Per the throughput model this is
   **−14% total wall** on the corpus box — it un-RAM-caps chr1/chr2 (49 GB → 30 GB → 4 workers
   fit instead of 2) *without* checkpointing's recompute. Stacks with int8 existing_h.
2. **Compute:** removes the **per-element `exp` (forward read) and `log` (forward store)** —
   the ~24% libm hotspot in `perf`. Read becomes a multiply-add; store becomes a divide; one
   `log` per locus instead of `N`. The backward's `exp(Alphamat+…)` reads likewise become
   direct `ahat` reads. (This *replaces* much of what PR #1's libmvec vectorizes — fewer
   transcendentals to vectorize at all.)
3. **Stability bonus:** no large offsets, no `exp(big − big)` cancellation — strictly better
   conditioned than today.

float also doubles SIMD width (AVX-512: 16 floats vs 8 doubles), compounding any residual
vectorized math.

---

## 4. Feasibility — proven numerically

- **Isolated mechanism demo** (`/tmp/offset_demo.py`): rescale-float32 error is `~1e-7` at
  `|offset|` from 0 to 2e6, while log-float32 grows to 6%. Rescale is robust to the exact thing
  that kills the drop-in.
- **Faithful recursion prototype** (forward + scaled backward, synthetic 1004 donors): rescale
  in **double** reproduces the double/log-space reference to 0.0000% (algorithm is correct),
  and rescale in **float32** stays exact.
- Remaining gate: the **full-pipeline chunklength Δ** on real chr11 (forward + the real backward
  chunk formulas), expected ~1e-6 (float relative) — well inside the calibration band, but must
  be measured, like every other numeric change here.

---

## 5. Scope of changes

- **`forwardAlgorithm`** — rewrite to store `ahat` (float) + `logScale`; drop the per-element
  `exp`/`log`. *Straightforward* (the math is in §2).
- **`backwardAlgorithm`** — (a) read `ahat` directly instead of `exp(Alphamat[t][i]+…-Alphasum)`;
  (b) rescale its own Beta recursion to linear + a backward log-scale; (c) **update the
  chunk-count / expected-chunk-length / mutation formulas** to use `ahat·bhat` with the
  scale-factor bookkeeping. *This is the delicate part* — the formulas mix forward, backward,
  and the normalizers (lines ~266–292), and the per-locus scales must cancel correctly.
  The Beta vectors are already O(Nhaps) (no big matrix), so they can stay double.
- **`InitialiseForward` / `sampler` signatures** — `ahat` is `float **`; add `logScale`.
- **`-s > 0` sampling path** — reads Alphamat by random access; either thread the scales through
  or keep that (rarely-used) path on the double/full matrix. Calibration uses `-s 0`.
- **Storage** — `float` Alphamat + `double logScale[T]` (T doubles ≈ 24 MB, negligible) and the
  per-locus `c[t]` already exist as `TransProb`.

---

## 6. Risk & effort

- **Not bit-identical** → 22-autosome chunkcount/chunklength re-validation (same gate as `-i`
  reduction, libmvec, checkpointing).
- **Forward rewrite:** ~1 day, low risk (textbook).
- **Backward + chunk formulas:** ~2–4 days, medium risk — the scale-cancellation in the
  expected-chunk-length terms is where bugs hide; prototype in Python/NumPy against the C output
  first.
- Lower risk than the parallel-scan idea (that's research; this is standard scaled FB).

---

## 7. Recommended path to revisit

1. Python/NumPy prototype of the **full** scaled forward–backward incl. the ChromoPainter
   chunk-count formulas; verify chunklengths match the C `double` build within tolerance on one
   chromosome.
2. Port forward → C (verify likelihood/EM unchanged).
3. Port backward + chunk formulas → C; verify chunk outputs.
4. Measure: RSS (expect ~half the Alphamat), wall (expect faster — fewer transcendentals),
   and chunklength Δ vs deployed across 22 autosomes.
5. PR.

---

## 8. Relationship to the other levers

- **Supersedes** the dead drop-in float; **delivers** the −14% the model attributes to float.
- **Subsumes** much of PR #1 (libmvec) on the forward — fewer transcendentals to vectorize.
- **Orthogonal to / better than checkpointing** on the corpus box (no recompute penalty); both
  can coexist (checkpoint the float `ahat` for extreme RAM limits).
- **Independent of** the parallel-scan idea (`docs/parallel-scan-li-stephens.md`), which targets
  the fork-join bound itself. Linear-rescale targets memory + transcendentals; parallel-scan
  targets the barrier. They compose.

---

## References

- Rabiner 1989, *Proc. IEEE* 77(2) — scaled forward–backward (§V.A), the basis here.
- Li & Stephens 2003, *Genetics* — the copying model.
- (Numerical evidence: `/tmp/offset_demo.py` and `/tmp/float_rescale_v2.py` from the design
  session; re-create from §1/§4.)
