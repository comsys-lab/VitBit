#!/bin/bash

# This script runs run1 executable twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

# ViT-Base with simultaneous execution of Tensor cores and FP CUDA cores.

# First run (warmup) - suppress output
echo "Warmup run..."
if [[ -x "TC_FP_ViT/TC_FP_ViT" ]]; then
    ./TC_FP_ViT/TC_FP_ViT > /dev/null 2>&1
else
    echo "Executable TC_FP_ViT/TC_FP_ViT not found or not executable."
fi

# Second run - display output
echo "Actual run..."
if [[ -x "TC_FP_ViT/TC_FP_ViT" ]]; then
    echo "TC_FP_ViT/TC_FP_ViT Result"
    ./TC_FP_ViT/TC_FP_ViT
    echo ""
else
    echo "Executable TC_FP_ViT/TC_FP_ViT not found or not executable."
fi