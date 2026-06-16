/* ChromoPainterFold.h - exact block-fold chunk-count (see ChromoPainterFold.c) */
#ifndef CHROMOPAINTERFOLD_H
#define CHROMOPAINTERFOLD_H

/* Compute per-population corrected chunk counts via the O(N*Umean) block fold,
   identical to the dense forward-backward-chunkcount. Deterministic fixed-param
   mode only (uniform copy_prob, global mut, no EM). Times the panel grouping
   build (*t_build) and the per-recipient fold (*t_fold) separately. */
void cpfold_perpop(int *newh, int **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *t_build, double *t_fold);

#endif
