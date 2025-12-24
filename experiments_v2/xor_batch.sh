#!/bin/bash

echo "========================================"
echo "XOR Experiments"
echo "Started: $(date)"
echo "========================================"
echo

for h in 3 4 5 6; do
    for m in 0.15 0.2 0.25 0.3 0.35; do
        echo "========================================"
        echo "Running: HEAD=$h, MUTATION=$m"
        echo "========================================"

        python xor.py --head $h --mutation $m --quiet

        echo
    done
done

echo "========================================"
echo "All experiments completed!"
echo "Finished: $(date)"
echo "========================================"