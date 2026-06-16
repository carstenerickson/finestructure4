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

# Files the fold produces (regional bootstrap is intentionally not folded).
EXTS="chunkcounts chunklengths mutationprobs"

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

echo "=== inline per-pop rel err (CPFOLD=1, -i 0, $ds) ==="
env CPFOLD=1 OMP_NUM_THREADS=1 "$FS" cp -g "$D/win.phase" -r "$D/win.recom" \
    -t "$ID" -f "$POP" 0 0 -s 0 -i 0 -n "$NE" -M "$MUT" -d -o "$TMP/b" 2>/dev/null \
    | grep "max rel err" | head -1

echo
if [ $fail -eq 0 ]; then echo "ALL EXACT (byte-identical at printed precision)"; else echo "SOME CONFIGS DIFFER - see FAIL lines above"; fi
exit $fail
