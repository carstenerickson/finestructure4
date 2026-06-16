#!/usr/bin/env bash
# validate_fold.sh - exactness gate for the linear-space dense (PR1) and the
# exact block fold (PR2) vs the upstream log-space dense.
#
# For each (dataset, mode) it runs four ChromoPainter configurations and asserts
# the standard output files are byte-identical at printed precision:
#   1. log-space dense   (CPLOG=1)            -- the upstream reference
#   2. linear-space dense (default, PR1)
#   3. -fold              (the block fold, PR2)
#   4. (inline) the CPFOLD=1 benchmark prints per-pop fold-vs-dense rel errs
#
# Usage:  ./validate_fold.sh [FS_BINARY] [DATADIR]
#   FS_BINARY defaults to ../fs ; DATADIR defaults to ./win
set -u
FS="${1:-../fs}"
D="${2:-./win}"
ID="$D/cp.idfile"; POP="$D/cp.poplist"
NE=400000; MUT=0.0006338578
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0

# Standard output files the fold reproduces byte-for-byte (regional bootstrap +
# samples are not folded). prop = copy proportions. EMprobs is checked separately
# (cmp_emprobs) because its N_e column is printed at %.10lf on a ~1e5 value, which
# exposes FP-reorder noise in the last digit - byte-identity there is not expected,
# but the SCHEMA (column count), the loglik columns and the mutation must match.
EXTS="chunkcounts chunklengths mutationprobs prop"

run() { # env_prefix phase recom extra_args outprefix
  env $1 OMP_NUM_THREADS=1 "$FS" cp -g "$2" -r "$3" -t "$ID" -f "$POP" 0 0 \
      -s 0 $4 -n "$NE" -M "$MUT" -o "$5" >/dev/null 2>&1
}

cmp_set() { # ref prefix label
  local ok=1
  for e in $EXTS; do
    [ -f "$1.$e.out" ] || continue
    if ! diff -q "$1.$e.out" "$2.$e.out" >/dev/null 2>&1; then ok=0; echo "    DIFFER: $e"; fi
  done
  if [ $ok -eq 1 ]; then echo "  PASS  $3"; else echo "  FAIL  $3"; fail=1; fi
}

# EMprobs comparator: enforce the SCHEMA (same column count per row - the fold
# regression was emitting 3 cols instead of 5) and that every field matches the
# reference, with a relative tolerance of 1e-6 on numeric fields (the N_e column
# carries FP-reorder noise in its last %.10lf digit across log/linear/fold).
cmp_emprobs() { # ref prefix label
  local r="$1.EMprobs.out" f="$2.EMprobs.out"
  [ -f "$r" ] && [ -f "$f" ] || { echo "  FAIL  $3 (EMprobs missing)"; fail=1; return; }
  if awk '
      function abs(x){return x<0?-x:x}
      FNR==NR{ nf[FNR]=NF; for(i=1;i<=NF;i++) v[FNR,i]=$i; refn=FNR; next }
      { candn=FNR;
        if(NF!=nf[FNR]){ bad=1 }                                  # schema: same column count
        for(i=1;i<=NF;i++){
          if($i ~ /[nN][aA][nN]|[iI][nN][fF]/ || v[FNR,i] ~ /[nN][aA][nN]|[iI][nN][fF]/){ bad=1 }  # reject NaN/Inf
          else if($i ~ /^-?[0-9.]+$/ && v[FNR,i] ~ /^-?[0-9.]+$/){
            d=abs(($i+0)-(v[FNR,i]+0)); rel=(abs(v[FNR,i]+0)>1?d/abs(v[FNR,i]+0):d);
            if(rel>1e-6) bad=1;
          } else if(v[FNR,i]!=$i){ bad=1 } } }
      END{ if(refn!=candn) bad=1; exit bad?1:0 }' "$r" "$f"; then   # same line count (no dropped rows)
    echo "  PASS  $3 (EMprobs schema + values, N_e tol 1e-6)"
  else echo "  FAIL  $3 (EMprobs schema/value/linecount mismatch)"; fail=1; fi
}

for ds in win200 win800 win win20k; do
  ph="$D/$ds.phase"; rc="$D/$ds.recom"
  [ -f "$ph" ] && [ -f "$rc" ] || continue
  echo "=== dataset $ds ($(sed -n 2p "$ph") SNPs) ==="
  for mode in "-i 0" "-i 6 -in -iM"; do
    run "CPLOG=1"  "$ph" "$rc" "$mode" "$TMP/log"
    run ""         "$ph" "$rc" "$mode" "$TMP/lin"
    run ""         "$ph" "$rc" "$mode -fold" "$TMP/fold"
    cmp_set "$TMP/log" "$TMP/lin"  "linear-dense == log-dense   [$mode]"
    cmp_set "$TMP/log" "$TMP/fold" "fold        == log-dense   [$mode]"
    cmp_emprobs "$TMP/log" "$TMP/lin"  "linear-dense == log-dense   [$mode]"
    cmp_emprobs "$TMP/log" "$TMP/fold" "fold        == log-dense   [$mode]"
  done
done

# --- per-pop fold exactness (PR2b): FIXED per-pop copy probs (-p) and/or FIXED
#     per-pop mutation rates (-m). The fold groups donors by (substring, pop),
#     so dense == fold at the per-pop output granularity. -m excludes -M. ---
runpp() { # poplist extra_args outprefix
  env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" \
      -t "$ID" -f "$1" 0 0 -s 0 -i 0 -n "$NE" $2 -o "$3" >/dev/null 2>&1
}
PR="$D/cp.poplist.prior"; PM="$D/cp.poplist.mut"; PB="$D/cp.poplist.pm"
echo "=== per-pop fold == dense (win, -i 0) ==="
if [ -f "$PR" ]; then
  runpp "$PR" "-M $MUT -p"        "$TMP/d_p";  runpp "$PR" "-M $MUT -p -fold"        "$TMP/f_p"
  cmp_set "$TMP/d_p"  "$TMP/f_p"  "fold == dense   [-p]"
fi
if [ -f "$PM" ]; then
  runpp "$PM" "-m $MUT"           "$TMP/d_m";  runpp "$PM" "-m $MUT -fold"           "$TMP/f_m"
  cmp_set "$TMP/d_m"  "$TMP/f_m"  "fold == dense   [-m]"
fi
if [ -f "$PB" ]; then
  runpp "$PB" "-p -m $MUT"        "$TMP/d_pm"; runpp "$PB" "-p -m $MUT -fold"        "$TMP/f_pm"
  cmp_set "$TMP/d_pm" "$TMP/f_pm" "fold == dense   [-p -m]"
fi

# --- all-vs-all (-a): -fold is exact here too (each individual is its own donor
#     pop, so the within-pop redistribution is a no-op). ---
echo "=== all-vs-all fold == dense (win, -a 0 0 -i 0) ==="
env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" -a 0 0 \
    -s 0 -i 0 -n "$NE" -M "$MUT"       -o "$TMP/d_a" >/dev/null 2>&1
env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" -a 0 0 \
    -s 0 -i 0 -n "$NE" -M "$MUT" -fold -o "$TMP/f_a" >/dev/null 2>&1
cmp_set "$TMP/d_a" "$TMP/f_a" "fold == dense   [-a 0 0]"

# --- output suppression: -fold must NOT write the regional bootstrap files (a
#     zero-filled .regionsquaredchunkcounts.out would collapse chromocombine's c)
#     and must REJECT the per-locus -b/-d outputs it cannot produce. ---
echo "=== -fold output suppression + per-locus outputs ==="
# regional bootstrap genuinely cannot be folded -> the files must be ABSENT.
env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" 0 0 \
    -s 0 -i 0 -k 5 -n "$NE" -M "$MUT" -fold -o "$TMP/sup" >/dev/null 2>&1
if [ -f "$TMP/sup.regionchunkcounts.out" ] || [ -f "$TMP/sup.regionsquaredchunkcounts.out" ]; then
  echo "  FAIL  regional files present under -fold"; fail=1
else echo "  PASS  regional files absent under -fold"; fi
# -b (.copyprobsperlocus, per-locus per-pop copy posterior) and -d (.transitionprobs,
# per-locus transition prob) ARE produced by the fold, byte-identical to the dense.
gzcmp() { # ext label  (compares <prefix>.ext.gz under $TMP/bd_d vs $TMP/bd_f)
  if diff <(gunzip -c "$TMP/bd_d.$1" 2>/dev/null) <(gunzip -c "$TMP/bd_f.$1" 2>/dev/null) >/dev/null 2>&1
  then echo "  PASS  $2"; else echo "  FAIL  $2"; fail=1; fi
}
for mode in "-i 0" "-i 6 -in -iM"; do
  env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" 0 0 \
      -s 0 $mode -n "$NE" -M "$MUT" -b -d       -o "$TMP/bd_d" >/dev/null 2>&1
  env OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" 0 0 \
      -s 0 $mode -n "$NE" -M "$MUT" -b -d -fold -o "$TMP/bd_f" >/dev/null 2>&1
  gzcmp copyprobsperlocus.out.gz "fold -b == dense   [$mode]"
  gzcmp transitionprobs.out.gz   "fold -d == dense   [$mode]"
done

# --- independent oracle: fs (dense AND fold) vs a from-scratch reference that
#     shares NO code with finestructure4. The reference (oracle/independent_oracle.py)
#     computes the per-pop chunk counts by exact brute-force enumeration of all 4^6
#     donor paths, cross-checked by a formula-free Monte Carlo. This catches a shared
#     bug in the emission / transition / chunk-count code that a fold-vs-dense diff
#     cannot (both share cp_emis). The case includes a missing allele (9). ---
OD="$(dirname "$D")/oracle"
if [ -f "$OD/data.phase" ]; then
  echo "=== independent oracle (brute-force + MC) vs fs dense and fold ==="
  EXP_A=1.4288035005; EXP_B=1.2117673272   # exact, from oracle/expected.txt
  orun() { # extra outprefix
    env OMP_NUM_THREADS=1 $1 "$FS" cp -g "$OD/data.phase" -r "$OD/data.recom" \
        -t "$OD/id.txt" -f "$OD/poplist.txt" 0 0 -j -s 0 -i 0 -n 100 -M 0.01 $2 -o "$3" >/dev/null 2>&1
  }
  ocheck() { # label chunkcounts-file
    awk -v ea="$EXP_A" -v eb="$EXP_B" -v lab="$2" '
      function abs(x){return x<0?-x:x}
      END{
        if(a=="" ){ print "  FAIL  "lab" (no output)"; exit 1 }
        da=abs(a-ea); db=abs(b-eb);
        if(da<1e-5 && db<1e-5) printf "  PASS  %s (popA %.6f popB %.6f, |d|<1e-5 vs exact)\n",lab,a,b;
        else { printf "  FAIL  %s popA=%.6f(d=%.1e) popB=%.6f(d=%.1e)\n",lab,a,da,b,db; exit 1 }
      }
      $1=="TGT"{a=$2;b=$3}' "$1"
  }
  orun "CPLOG=1" ""      "$TMP/odlog"; ocheck "$TMP/odlog.chunkcounts.out" "log-dense  == oracle" || fail=1
  orun ""        ""      "$TMP/odlin"; ocheck "$TMP/odlin.chunkcounts.out" "lin-dense  == oracle" || fail=1
  orun ""        "-fold" "$TMP/ofold"; ocheck "$TMP/ofold.chunkcounts.out" "fold       == oracle" || fail=1
fi

# --- path-sampling consistency (PR1): the linear-space default must produce the
#     SAME samples as the log-space reference. forwardAlgorithmLin fills Alphamat
#     LINEAR but the sampler reads it LOG-space, so the FINAL (sampling) run falls
#     back to the log forward; E-M iterations stay linear (the speedup). Fixed RNG
#     seed (-S) makes samples deterministic, so default == CPLOG byte-for-byte.
#     Two cases: -i 0 (single run is the final/sampling run) and -i 4 -in -iM
#     (E-M iterations linear, only the final run log -> the per-run path). ---
echo "=== sampling: linear-default == log (fixed seed, win) ==="
samp() { # env_prefix iters outprefix
  env $1 OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" -t "$ID" -f "$POP" 0 0 \
      -s 5 -S 1 $2 -n "$NE" -M "$MUT" -o "$3" >/dev/null 2>&1
  gunzip -f "$3.samples.out.gz" 2>/dev/null
}
samp_cmp() { # iters label
  samp "CPLOG=1" "$1" "$TMP/slog"; samp "" "$1" "$TMP/slin"
  if [ -f "$TMP/slog.samples.out" ] && [ -f "$TMP/slin.samples.out" ]; then
    if diff -q "$TMP/slog.samples.out" "$TMP/slin.samples.out" >/dev/null 2>&1; then
      echo "  PASS  samples linear-default == log   [$2]"
    else echo "  FAIL  samples linear-default != log   [$2]"; fail=1; fi
  else echo "  FAIL  samples not produced   [$2]"; fail=1; fi
}
samp_cmp "-i 0"              "single sampling run"
samp_cmp "-i 4 -in -iM"     "linear E-M + log final run"

# Full-precision per-pop rel err: only available when fs is built with the dev
# benchmark, i.e. -DCP_FOLD_BENCH (the CPFOLD=1 scaffolding is excluded from
# release builds). Absent => this prints the note; the byte-identity gate above
# is the real check and does not need it.
echo "=== inline per-pop rel err (CPFOLD=1, requires -DCP_FOLD_BENCH build) ==="
relerr=$(env CPFOLD=1 OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" \
    -t "$ID" -f "$POP" 0 0 -s 0 -i 0 -n "$NE" -M "$MUT" -d -o "$TMP/b" 2>/dev/null \
    | grep "max rel err" | head -1)
[ -n "$relerr" ] && echo "$relerr" || echo "  (skipped: fs not built with -DCP_FOLD_BENCH)"

echo
if [ $fail -eq 0 ]; then echo "ALL EXACT (byte-identical at printed precision)"; else echo "SOME CONFIGS DIFFER - see FAIL lines above"; fi
exit $fail
