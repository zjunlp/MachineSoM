#!/bin/bash

# 0,1 2,3 4,5 6,7 8,9
experiment_type="turn"
which_turn=-1
turn=4
# repeat=(1 2 3 4 5 6 7)
# dataset=("mmlu" "chess")
# repeat=(1 2 3 4 5)
# dataset=("math")
# dataset=("mmlu")
# repeat=(2 3 4 5 7)
dataset=("chess")
repeat=(2 3 4 5 6)


for r in "${repeat[@]}"; do
    for ds in "${dataset[@]}"; do
        # python your_python_script.py --a "$a" --b "$b"
        echo "$r, $ds"
        python evaluate.py --dataset "$ds" --metric "acc" --repeat "$r" --which_turn "$which_turn" --experiment_type "$experiment_type" --role '[1]' --turn "$turn"
    done
done
