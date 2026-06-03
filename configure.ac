AC_INIT([fs], [4.1.0], [dan.lawson@bristol.ac.uk],
             [fs], [http://www.paintmychromosomes.com])
AC_PREREQ([2.59])
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([foreign 1.10 -Wall no-define])
AC_CONFIG_HEADERS([config.h])
AC_LANG(C++)
AC_PROG_CXX
AC_PROG_CC # Perhaps causes errors in some places, but should be fine

LT_INIT
dnl Required for the noinst_LIBRARIES (libcppaint.a) below; without it
dnl Automake 1.12+ warns and fails under -Werror on the first non-libtool ar.
AM_PROG_AR
AC_CONFIG_MACRO_DIR([m4])
#AM_PROG_CC_C_O
#AX_CXX_COMPILE_STDCXX_11([noext], [mandatory])

# Checks for libraries.
# find gsl, wherever it's hiding on the system
#AC_CHECK_LIB([m],[cos])
#AC_CHECK_LIB([gslcblas],[cblas_dgemm])
#AC_CHECK_LIB([gsl],[gsl_blas_dgemm])
#AX_PATH_GSL(,,AC_MSG_ERROR(could not find required version of GSL))
#AC_CHECK_LIB([z],[gzread],,AC_MSG_ERROR(Could not find zlib))

AC_OPENMP
AC_SUBST(AM_CXXFLAGS,"$OPENMP_CXXFLAGS")

dnl Widest AVX -march the C compiler accepts, for the libmvec-vectorized
dnl painter TU (libcppaint.a). The exp/log vector clones need >= AVX2.
dnl Falls back to empty (scalar libm, no SIGILL risk) on older compilers
dnl that reject the token, or on non-x86 hosts -- the build still succeeds.
AC_LANG_PUSH([C])
PAINTER_MARCH=
for fs_cand in "-march=x86-64-v3" "-march=core-avx2" "-mavx2"; do
  fs_save_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS $fs_cand"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],[PAINTER_MARCH="$fs_cand"])
  CFLAGS="$fs_save_CFLAGS"
  test -n "$PAINTER_MARCH" && break
done
AC_LANG_POP([C])
AC_MSG_NOTICE([painter vectorization -march: ${PAINTER_MARCH:-none (scalar libm)}])
AC_SUBST([PAINTER_MARCH])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
