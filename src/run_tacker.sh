#!/bin/bash

# This script runs run1 executable twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

# ViT-Base with simultaneous execution of Tensor cores and INT CUDA cores.

# First run (warmup) - suppress output
echo "Warmup run..."
if [[ -x "TC_INT_ViT/TC_INT_ViT" ]]; then
    ./TC_INT_ViT/TC_INT_ViT > /dev/null 2>&1
else
    echo "Executable TC_INT_ViT/TC_INT_ViT not found or not executable."
fi

# Second run - display output
echo "Actual run..."
if [[ -x "TC_INT_ViT/TC_INT_ViT" ]]; then
    echo "TC_INT_ViT/TC_INT_ViT Result"
    ./TC_INT_ViT/TC_INT_ViT
    echo ""
else
    echo "Executable TC_INT_ViT/TC_INT_ViT not found or not executable."
fi