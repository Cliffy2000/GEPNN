#!/bin/bash

echo "T-XOR Experiments"
echo "Started: $(date)"
echo

T1=0
T2=-2

for h in 3 4 5 6 7; do
    for m in 0.15 0.2 0.25 0.3 0.35; do
        echo "============================================================"
        echo "Running: HEAD=$h, MUTATION=$m, T1=$T1, T2=$T2"
        python txor_sync.py --head $h --mutation $m --t1 $T1 --t2 $T2 --quiet
        echo
        echo
    done
done

echo
echo
echo "============================================================"
echo
python ../reports_v2/txor_results.py --t1 $T1 --t2 $T2
echo
echo "============================================================"
echo

echo "Finished: $(date)"