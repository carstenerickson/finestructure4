/* ChromoPainterFold.h - exact block-fold chunk-count (see ChromoPainterFold.c) */
#ifndef CHROMOPAINTERFOLD_H
#define CHROMOPAINTERFOLD_H

/* Compute per-population corrected chunk counts, expected differences, expected
   chunk lengths and N_e via the O(N*Umean) block fold, identical to the dense
   forward-backward-chunkcount. Donors are grouped by (local substring, donor
   population), so this is exact for UNIFORM copy_prob + global mutation and for
   FIXED per-population copy_prob (copy_prob[]) and per-donor mutation rates
   (MutProb_vec[]). It does not run per-pop E-M updates. Times the panel grouping
   build (*t_build) and the per-recipient fold (*t_fold) separately. */
void cpfold_perpop(int *newh, int **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob, double *copy_probSTART,
                   double *pos, double *lambda, double delta, double rhobar,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *out_ndiff, double *out_nlen, double *out_Ne,
                   double *t_build, double *t_fold);

#endif
