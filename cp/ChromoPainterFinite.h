#ifndef CHROMOPAINTERFINITE_H
#define CHROMOPAINTERFINITE_H

/* cp_bad_value: 1 if x is NaN or +/-Inf, else 0.

   Deliberately defined in its own translation unit (ChromoPainterFinite.c),
   which is compiled WITHOUT -ffast-math. ChromoPainterSampler.c is built
   with -ffast-math (-> -ffinite-math-only), under which the optimizer is
   licensed to assume no NaN/Inf and fold an inline isnan()/bit-test to a
   constant 0 -- silently disabling the likelihood sanity guard. Keeping the
   test in a separate, non-fast-math TU (and not LTO-inlined) preserves it.
   Detects the whole exponent-all-ones class, so it catches +/-Inf too, which
   -ffinite-math-only makes a likely failure mode (e.g. log(0) -> -Inf). */
int cp_bad_value(double x);

#endif
