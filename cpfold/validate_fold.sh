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

echo "=== inline per-pop rel err (CPFOLD=1, -i 0, $ds) ==="
env CPFOLD=1 OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" \
    -t "$ID" -f "$POP" 0 0 -s 0 -i 0 -n "$NE" -M "$MUT" -d -o "$TMP/b" 2>/dev/null \
    | grep "max rel err" | head -1

echo
if [ $fail -eq 0 ]; then echo "ALL EXACT (byte-identical at printed precision)"; else echo "SOME CONFIGS DIFFER - see FAIL lines above"; fi
exit $fail
