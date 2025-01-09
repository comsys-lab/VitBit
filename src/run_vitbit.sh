#!/bin/bash

# This script runs run1 executable twice
# The first run is for warmup and the output is suppressed
# The second run displays the output

# ViT-Base with using VitBit

# First run (warmup) - suppress output
echo "Warmup run..."
if [[ -x "VitBit/VitBit" ]]; then
    ./VitBit/VitBit > /dev/null 2>&1
else
    echo "Executable VitBit/VitBit not found or not executable."
fi

# Second run - display output
echo "Actual run..."
if [[ -x "VitBit/VitBit" ]]; then
    echo "VitBit/VitBit Result"
    ./VitBit/VitBit
    echo ""
else
    echo "Executable VitBit/VitBit not found or not executable."
fi