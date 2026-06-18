/* ChromoPainterFold.h - exact block-fold chunk-count (see ChromoPainterFold.c) */
#ifndef CHROMOPAINTERFOLD_H
#define CHROMOPAINTERFOLD_H

/* Numeric constants and the emission function shared by the dense forward-
   backward (ChromoPainterSampler.c) and the exact block fold (ChromoPainterFold.c),
   so the two cannot drift apart - they must produce identical output. */
#ifndef SMALL_NUM
#define SMALL_NUM 1e-20   /* emission floor for the gap allele (code 8) */
#endif
#ifndef MIN_NE
#define MIN_NE 1e-8       /* lower floor on the N_e (-in) E-M estimate */
#endif

/* emission e(recipient allele r, donor allele d, donor mutation m):
   r==9 -> 1.0 (recipient missing ignored); r==8 -> gap (SMALL_NUM); else (1-m)/m. */
static inline double cp_emis(int r, int d, double m){
    if(r==9) return 1.0;
    if(r==8) return (r==d)?(1-SMALL_NUM):SMALL_NUM;
    return (r==d)?(1-m):m;
}

/* Compute per-population corrected chunk counts, expected differences, expected
   chunk lengths, N_e and the forward log-likelihood (*out_loglik, the same value
   the dense forward returns) via the O(N*Umean) block fold, identical to the dense
   forward-backward-chunkcount. Donors are grouped by (local substring, donor
   population), so this is exact for UNIFORM copy_prob + global mutation and for
   FIXED per-population copy_prob (copy_prob[]) and per-population mutation rates
   (MutProb_vec[]). Times the panel grouping build (*t_build) and the per-recipient
   fold (*t_fold) separately.

   out_ccpop[p] is the per-pop corrected chunk count INCLUDING the start term;
   out_start[p] (optional, may be NULL) is the per-pop start-term contribution
   alone (= sum over donors in pop p of the locus-0 posterior). out_ccpop-out_start
   is therefore the per-pop posterior chunk count EXCLUDING start, the dense
   copy_prob_new per-pop total that drives the copy-proportion E-M update (-ip). */
/* retain_panel != 0 (single-recipient -fold): the locus-major donor buffer is built
   once and kept across calls with the same (nhaps,nloci) panel, so existing_h is read
   only on the first call. On that first build each hap-major donor row is freed (via the
   existing_h alias) as soon as it is copied - before the fold working set is allocated -
   so the two never coexist and the peak drops; existing_h[i] is nulled. The caller must
   then null the canonical all_chromosomes entries (they alias the freed rows) so cleanup
   does not double-free. Pass 0 to rebuild-and-free the buffer per call without touching
   existing_h (the default, required when the panel changes between recipients). Call
   cpfold_cleanup() once at teardown to release a retained buffer. */
/* Regional bootstrap (optional, all NULL/0 to skip): with out_regfinal/out_regsq
   (npop) and out_numreg, the fold replicates the dense's per-region chunk-count
   banking - the cumulative per-locus total is banked into a region every time it
   crosses region_size (rounding 1e-7), the per-pop sums and sums-of-squares
   accumulate into out_regfinal/out_regsq, and out_numreg counts the regions. The
   final partial region is dropped, matching the dense. FP-equivalent (~1e-9) to the
   dense, NOT bit-identical: a boundary can land one locus off the dense, so the raw
   per-region files may differ - the conserved quantities are the aggregate sum and
   chromocombine's c. */
void cpfold_perpop(signed char *newh, signed char **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob, double *copy_probSTART,
                   double *pos, double *lambda, double delta, double rhobar,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *out_start, double *out_ndiff, double *out_nlen, double *out_Ne,
                   double *out_loglik, double *out_etp, double *out_ecp,
                   double region_size, double *out_regfinal, double *out_regsq, int *out_numreg,
                   int samplesTOT, int *out_samples,
                   double *t_build, double *t_fold, int retain_panel);

/* Free the retained locus-major donor buffer (no-op if none). Idempotent. */
void cpfold_cleanup(void);

#endif
