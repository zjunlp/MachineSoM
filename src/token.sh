#!/bin/bash

# dataset="mmlu"
# repeat="[2,3,4,5,7]"

dataset="chess"
repeat="[2,3,4,5,6]"

role=(0 1 2 3)
strategy=("000" "001" "010" "011" "100" "101" "110" "111")

for r in "${role[@]}"; do
    python tokens.py --repeat "$repeat" --dataset "$dataset" --role "[$r]" | grep "total_tokens"
done

for s in "${strategy[@]}"; do
    python tokens.py --repeat "$repeat" --dataset "$dataset" --strategy "['$s']" | grep "total_tokens"
done
