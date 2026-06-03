AC_INIT([fs], [4.1.0], [dan.lawson@bristol.ac.uk],
             [fs], [http://www.paintmychromosomes.com])
AC_PREREQ([2.59])
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([foreign 1.10 -Wall no-define])
AC_CONFIG_HEADERS([config.h])
AC_LANG(C++)
AC_PROG_CXX
AC_PROG_CC # Perhaps causes errors in some places, but should be fine

AC_CANONICAL_HOST
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

dnl --- ChromoPainter painter-TU acceleration --------------------------------
dnl Default: a single PORTABLE binary plus runtime AVX2 dispatch. The hot
dnl forward/backward routines are built with function multi-versioning
dnl (target_clones); a glibc IFUNC resolver selects an AVX2 clone (the
dnl vectorizer emits libmvec exp/log there under -ffast-math) at load time on
dnl capable CPUs, while the binary still runs on any baseline x86-64. If the
dnl toolchain can't do target_clones + IFUNC (old GCC, non-ELF, non-x86), we
dnl fall back to a portable scalar build -- correct, just not accelerated.
dnl --enable-native instead tunes the whole painter TU to the BUILD host
dnl (-march=native): faster when build host == run host, but the binary may
dnl SIGILL on older CPUs, so it is opt-in and disables FMV.
AC_ARG_ENABLE([native],
  [AS_HELP_STRING([--enable-native],
     [tune the ChromoPainter TU to the build CPU (-march=native); faster but not
      portable to older CPUs. Default: portable binary + runtime AVX2 dispatch.])],
  [fs_native=$enableval], [fs_native=no])

PAINTER_MARCH=
PAINTER_FMV=
AC_LANG_PUSH([C])
AS_IF([test "x$fs_native" = xyes],
  [fs_save_CFLAGS="$CFLAGS"; CFLAGS="$CFLAGS -march=native"
   AC_MSG_CHECKING([whether the C compiler accepts -march=native])
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],
     [PAINTER_MARCH="-march=native"; AC_MSG_RESULT([yes])],
     [AC_MSG_RESULT([no; building portable])])
   CFLAGS="$fs_save_CFLAGS"],
  [AS_CASE([$host_cpu],
     [i?86|x86_64|amd64],
       [AC_MSG_CHECKING([for __attribute__((target_clones)) with IFUNC dispatch])
        AC_LINK_IFELSE(
          [AC_LANG_SOURCE([[
__attribute__((target_clones("avx2","default")))
int fs_fmv_probe(void){ return 1; }
int main(void){ return fs_fmv_probe() - 1; }
]])],
          [PAINTER_FMV="-DCP_USE_FMV=1"; AC_MSG_RESULT([yes])],
          [AC_MSG_RESULT([no; portable scalar build])])],
     [AC_MSG_NOTICE([non-x86 host ($host_cpu): painter built portable (scalar libm)])])])
AC_LANG_POP([C])
AS_IF([test -n "$PAINTER_MARCH"],
  [AC_MSG_NOTICE([painter acceleration: -march=native, FMV off (binary not portable)])],
  [AS_IF([test -n "$PAINTER_FMV"],
     [AC_MSG_NOTICE([painter acceleration: portable binary + runtime AVX2 dispatch (FMV)])],
     [AC_MSG_NOTICE([painter acceleration: portable scalar (no FMV, no -march)])])])
AC_SUBST([PAINTER_MARCH])
AC_SUBST([PAINTER_FMV])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
