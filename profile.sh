#!/bin/bash
# Profile one variant with Nsight Compute (full metric set).
#   ./profile.sh vitbit       ->  profiles/vitbit.ncu-rep
# Variants: tc tacker tc_fc tc_ic_fc vitbit. Needs root or GPU profiling permission.

v=$1
if [[ -z $v || ! -x bin/$v ]]; then
    echo "usage: $0 <tc|tacker|tc_fc|tc_ic_fc|vitbit>   (build first with 'make')"; exit 1
fi
mkdir -p profiles
ncu --set full -o profiles/$v bin/$v
