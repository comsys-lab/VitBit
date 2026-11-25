#!/bin/bash
# Run ViT-Base inference and print the inference time of each variant.
#   ./run.sh            # all variants, then the speedup over TC
#   ./run.sh vitbit     # one or more variants
# Each variant runs once for warm-up (output discarded) and once for measurement.
# Build first with `make`.

ALL="tc tacker tc_fc tc_ic_fc vitbit"
VARIANTS=${@:-$ALL}

declare -A TIME
for v in $VARIANTS; do
    exe=bin/$v
    if [[ ! -x $exe ]]; then
        echo "$exe not found (run 'make $v' first)"; exit 1
    fi
    $exe > /dev/null 2>&1                      # warm-up
    out=$($exe)                                # measured run
    t=$(echo "$out" | sed -n 's/^Inference Time: \([0-9.]*\)ms/\1/p')
    TIME[$v]=$t
    printf "%-9s %10s ms\n" "$v" "$t"
done

# Speedup over the Tensor-core-only baseline
if [[ -n ${TIME[tc]} && $# -eq 0 ]]; then
    echo
    echo "Speedup over TC:"
    for v in $VARIANTS; do
        awk -v b="${TIME[tc]}" -v t="${TIME[$v]}" -v n="$v" 'BEGIN { printf "%-9s %6.2fx\n", n, b / t }'
    done
fi
