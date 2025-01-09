#!/bin/bash

# This script runs all compiled CUDA executables twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

echo "End-to-End Inference Time Comparison..."

# List of executable files
executables=(
    "TC_ViT/TC_ViT"
    "TC_INT_ViT/TC_INT_ViT"
    "TC_FP_ViT/TC_FP_ViT"
    "TC_INT_FP_ViT/TC_INT_FP_ViT"
    "VitBit/VitBit"
)

# First run (warmup) - suppress output
echo "Warmup run..."
for exe in "${executables[@]}"
do
    if [[ -x "$exe" ]]; then
        ./"$exe" > /dev/null 2>&1
    else
        echo "Executable $exe not found or not executable."
    fi
done

# Loop through each executable and run it
echo "Actual run..."
for exe in "${executables[@]}"
do
    if [[ -x "$exe" ]]; then
        echo "$exe Result"
        ./"$exe"
        echo ""
    else
        echo "Executable $exe not found or not executable."
    fi
done