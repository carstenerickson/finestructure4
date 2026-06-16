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
   (MutProb_vec[]). It does not run per-pop E-M updates. Times the panel grouping
   build (*t_build) and the per-recipient fold (*t_fold) separately. */
void cpfold_perpop(int *newh, int **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob, double *copy_probSTART,
                   double *pos, double *lambda, double delta, double rhobar,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *out_ndiff, double *out_nlen, double *out_Ne,
                   double *out_loglik, double *out_etp, double *out_ecp, double *t_build, double *t_fold);

#endif
