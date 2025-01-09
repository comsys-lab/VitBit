#!/bin/bash

# This script runs run1 executable twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

# ViT-Base with simultaneous execution of  Tensor, INT, and FP CUDA cores.

# First run (warmup) - suppress output
echo "Warmup run..."
if [[ -x "TC_INT_FP_ViT/TC_INT_FP_ViT" ]]; then
    ./TC_INT_FP_ViT/TC_INT_FP_ViT > /dev/null 2>&1
else
    echo "Executable TC_INT_FP_ViT/TC_INT_FP_ViT not found or not executable."
fi

# Second run - display output
echo "Actual run..."
if [[ -x "TC_INT_FP_ViT/TC_INT_FP_ViT" ]]; then
    echo "TC_INT_FP_ViT/TC_INT_FP_ViT Result"
    ./TC_INT_FP_ViT/TC_INT_FP_ViT
    echo ""
else
    echo "Executable TC_INT_FP_ViT/TC_INT_FP_ViT not found or not executable."
fi