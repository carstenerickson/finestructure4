#include "ChromoPainterFinite.h"

/* See ChromoPainterFinite.h. Must NOT be compiled with -ffast-math /
   -ffinite-math-only, or the compiler may assume the argument is finite and
   fold this to `return 0`. The bit test reads the IEEE-754 exponent field
   directly: all-ones exponent => NaN (nonzero mantissa) or +/-Inf (zero
   mantissa). */
int cp_bad_value(double x){
  union { double d; unsigned long long u; } v;
  v.d = x;
  return ((v.u & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL);
}
