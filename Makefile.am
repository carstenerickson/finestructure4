## CC = /opt/homebrew/opt/llvm/bin/clang
## CXX = $(CC)++

ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = subdir-objects

bin_PROGRAMS = fs mixPainter

##OPTIMIZATION = -O3 -lm -Wall -Wno-write-strings -mfpmath=sse -msse -msse2 -funroll-loops -fomit-frame-pointer -ftree-vectorize -funsafe-math-optimizations -fopenmp -Wno-unused-variable $(OPENMP_CXXFLAGS)
OPTIMIZATION = -O3 -Wall -Wno-write-strings -funroll-loops -fomit-frame-pointer -ftree-vectorize -funsafe-math-optimizations -Wno-unused-variable -fopenmp $(OPENMP_CXXFLAGS)
## For mac: may need to add -Wl,-ld_classic
## For others: may need to remove them?
fs_CXXFLAGS = $(OPTIMIZATION) -Wall -Wl,-ld_classic ##  -fsanitize=leak
## The ChromoPainter inner-loop OpenMP pragmas live in *.c files, so the
## same -fopenmp + -O3 + vectorize flags need to flow into the C compile
## rule, not just the C++ one. Without fs_CFLAGS, the .c sources compile
## with the autoconf default `CFLAGS = -g -O2` (no -fopenmp, no -O3), and
## the pragmas are silently no-ops. Mirroring fs_CXXFLAGS keeps the two
## languages in lockstep.
fs_CFLAGS = $(OPTIMIZATION) -Wall ##  -fsanitize=leak
fs_LDFLAGS = $(OPTIMIZATION) -lm -static-libstdc++

mixPainter_CXXFLAGS = $(OPTIMIZATION) -Wall
mixPainter_CFLAGS = $(OPTIMIZATION) -Wall
## -lm: the sampler's exp/log live in libcppaint.a (linked below); on libcs
## where libm isn't folded into libc, or under strict --as-needed, mixPainter
## must request it explicitly (fs_LDFLAGS already does).
mixPainter_LDFLAGS = $(OPTIMIZATION) -static-libstdc++ -lm

all_SOURCES = fsproject.h fsproject.cpp fssettings.h fssettings.cpp fsutils.h fsutils.cpp fsparam.h fsparam.cpp fscmds.h fscmds.cpp fsdonor.cpp fsdonor.h finestructure/fines.h alphanum.hpp fsconstants.h cp/ChromoPainterReading.h cp/ChromoPainterConstants.h cp/ChromoPainterError.h cp/ChromoPainterData.h  cp/ChromoPainterMutEM.h cp/ChromoPainterPar.h cp/ChromoPainterRecmap.h cp/ChromoPainterDonors.h cp/ChromoPainterOutfiles.h cp/ChromoPainterInfiles.h cp/ChromoPainterSampler.h cp/ChromoPainterFinite.h cp/ChromoPainterFinite.c cp/ChromoPainterReading.c cp/ChromoPainterError.c cp/ChromoPainterData.c  cp/ChromoPainterMutEM.c cp/ChromoPainterPar.c cp/ChromoPainterRecmap.c cp/ChromoPainterDonors.c cp/ChromoPainterOutfiles.c cp/ChromoPainterInfiles.c cp/adler32.c cp/compress.c cp/crc32.c cp/crc32.h cp/deflate.c cp/deflate.h cp/gzclose.c cp/gzguts.h cp/gzlib.c cp/gzread.c cp/gzwrite.c cp/infback.c cp/inffast.c cp/inffast.h cp/inffixed.h cp/inflate.c cp/inflate.h cp/inftrees.c cp/inftrees.h cp/trees.c cp/trees.h cp/uncompr.c cp/zconf.h cp/zlib.h cp/zutil.c cp/zutil.h finestructure/safegetline.h finestructure/data.h finestructure/fsxml.h finestructure/state.h finestructure/prior.h finestructure/node.h finestructure/inf1.h finestructure/infadmixture.h finestructure/infconcordance.h  finestructure/infextract.h finestructure/infextract2.h finestructure/infextract3.h finestructure/infextract4.h finestructure/infextract5.h finestructure/infextractdonors.h finestructure/infmcmc.h finestructure/fines.h finestructure/finesfunctions.h finestructure/rng.h finestructure/safegetline.cpp finestructure/data.cpp finestructure/fsxml.cpp finestructure/state.cpp finestructure/prior.cpp finestructure/node.cpp finestructure/inf1.cpp finestructure/infadmixture.cpp finestructure/infconcordance.cpp finestructure/infextract.cpp finestructure/infextract2.cpp finestructure/infextract3.cpp finestructure/infextract4.cpp finestructure/infextract5.cpp finestructure/infextractdonors.cpp finestructure/infmcmc.cpp finestructure/finesfunctions.cpp finestructure/fines.cpp finestructure/rng.cpp chromocombine/ChromoCombine.cpp chromocombine/ChromoCombineFuns.cpp chromocombine/ChromoCombineFuns.h chromocombine/ChromoCombine.h
fs_SOURCES = fs.cpp $(all_SOURCES)
mixPainter_SOURCES = badpainter.cpp $(all_SOURCES)

## ChromoPainterSampler.c is compiled here with -ffast-math so the vectorizer
## can emit libmvec exp()/log() for the per-locus inner loops: ~12% faster at
## 4 threads (chr11 EUR panel). Output is identical at the printed
## (calibration) precision -- chunkcounts/chunklengths md5 match the
## non-fast-math build. (Raw doubles already vary ~1e-13 across thread counts
## on this base from the OpenMP reduction combine order; this PR doesn't change
## that, and stacks cleanly on the deterministic-sum PR, which makes it exact.)
##
## Scoped to this single TU via a convenience library so the rest of fs (e.g.
## finestructure inference) is untouched. NOTE: $(OPTIMIZATION) already enables
## -funsafe-math-optimizations everywhere, so the baseline is not strict IEEE;
## the delta here is the rest of -ffast-math -- chiefly -ffinite-math-only,
## which is what lets the vectorizer call the libmvec clones. The NaN/Inf
## likelihood guard lives in a separate non-fast-math TU
## (cp/ChromoPainterFinite.c: cp_bad_value) so -ffinite-math-only can't fold it
## away.
##
## CPU targeting (see configure.ac): @PAINTER_FMV@ is -DCP_USE_FMV when the
## toolchain supports function multi-versioning -- the hot forward/backward
## routines then get a baseline + AVX2 clone with load-time IFUNC dispatch, so
## one PORTABLE binary runs anywhere yet takes the AVX2 (libmvec) path on
## capable CPUs. No TU-wide -march, so nothing here can SIGILL on an old CPU.
## @PAINTER_MARCH@ is empty by default and becomes -march=native only under
## ./configure --enable-native (build host == run host; not portable).
noinst_LIBRARIES = libcppaint.a
libcppaint_a_SOURCES = cp/ChromoPainterSampler.c
libcppaint_a_CFLAGS = $(OPTIMIZATION) -ffast-math @PAINTER_MARCH@ @PAINTER_FMV@
fs_LDADD = libcppaint.a
mixPainter_LDADD = libcppaint.a

dist_noinst_SCRIPTS = scripts/finestructuregreedy.sh scripts/qsub_array.sh scripts/sbatch_array.sh scripts/makeuniformrecfile.pl scripts/convertrecfile.pl scripts/idsFromChunkcounts.sh scripts/phasescreen.pl scripts/phasesubsample.pl scripts/impute2chromopainter.pl scripts/beagle2chromopainter.pl scripts/msms2cp.pl Manual/manual.pdf examples/example1 examples/example2 examples/example3 examples/example4 LICENSE-FS LICENSE-CP
