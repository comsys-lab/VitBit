#!/bin/bash

VITBIT_EXECUTABLE="../TC_ViT/TC_ViT"

PROFILING_RESULT="./profile_TC_ViT"

ncu -o $PROFILING_RESULT --set full $VITBIT_EXECUTABLE