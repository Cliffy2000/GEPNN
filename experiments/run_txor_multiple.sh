#!/bin/bash

SCRIPT_PATH=txor.py
LOG_FILE=txor_multiple_results.txt

# Clear log file and write header
echo "XOR Experiment Results" > $LOG_FILE
echo "======================" >> $LOG_FILE
echo "Started: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "HEAD_LENGTH, MUTATION_RATE, PERFECT_COUNT" >> $LOG_FILE

for h in 9; do
    for m in 0.25 0.3 0.35 0.4; do
        echo "========================================"
        echo "Running: HEAD_LENGTH=$h, MUTATION_RATE=$m"
        echo "========================================"

        # Modify HEAD_LENGTH
        sed -i "s/^HEAD_LENGTH = .*/HEAD_LENGTH = $h/" $SCRIPT_PATH

        # Modify MUTATION_RATE
        sed -i "s/^MUTATION_RATE = .*/MUTATION_RATE = $m/" $SCRIPT_PATH

        # Run the experiment and capture output
        OUTPUT=$(python $SCRIPT_PATH | grep "perfect solutions")

        # Extract perfect count from output like "Completed: 95/100 perfect solutions"
        PERFECT=$(echo $OUTPUT | sed 's/.*: \([0-9]*\)\/.*/\1/')

        # Log result
        echo "$h, $m, $PERFECT" >> $LOG_FILE
        echo "Result: HEAD_LENGTH=$h, MUTATION_RATE=$m, PERFECT=$PERFECT"
        echo ""
    done
done

echo "" >> $LOG_FILE
echo "Completed: $(date)" >> $LOG_FILE

echo "========================================"
echo "All experiments completed!"
echo "Results saved to $LOG_FILE"
echo "========================================"
cat $LOG_FILE