#!/bin/bash

# This script runs run1 executable twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

# ViT-Base with only utilizing Tensor cores.

# First run (warmup) - suppress output
echo "Warmup run..."
if [[ -x "TC_ViT/TC_ViT" ]]; then
    ./TC_ViT/TC_ViT > /dev/null 2>&1
else
    echo "Executable TC_ViT/TC_ViT not found or not executable."
fi

# Second run - display output
echo "Actual run..."
if [[ -x "TC_ViT/TC_ViT" ]]; then
    echo "TC_ViT/TC_ViT Result"
    ./TC_ViT/TC_ViT
    echo ""
else
    echo "Executable TC_ViT/TC_ViT not found or not executable."
fi