#!/bin/bash

VITBIT_EXECUTABLE="../VitBit/VitBit"

PROFILING_RESULT="./profile_VitBit"

ncu -o $PROFILING_RESULT --set full $VITBIT_EXECUTABLE