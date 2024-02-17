#!/bin/bash
DATASET="chess"
TURN=3
MODEL="AnyscaleMixtral"
API_ACCOUNT="anyscale"
API_IDX=0
lists=(
    "eo ee oo"
    "eoe eoo eee ooo"
    "eoee eooo eoeo eeee oooo"
    "eoeee eoooo eoeoe eeeee ooooo"
    "eoeeee eooooo eoeoeo eeeeee oooooo"
    "eoeeeee eoooooo eoeoeoe eeeeeee ooooooo"
    "eoeeeeee eooooooo eoeoeoeo eeeeeeee oooooooo"
    "eoeeeeeee eoooooooo eoeoeoeoe eeeeeeeee ooooooooo"
    "eoeeeeeeee eooooooooo eoeoeoeoeo eeeeeeeeee oooooooooo"
)
OUTPUT_DIR="log-agent"
mkdir -p $OUTPUT_DIR

for REPEAT in {1..5}
do
    mkdir -p "results/$EXP_TYPE/$DATASET/$REPEAT"
done

for AGENT in {2..10}
do
    COUNT=0
    for CASE_ID in "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"
    do
        index=$((AGENT - 2))
        IFS=' ' read -r -a elements <<< "${lists[$index]}"
        for element in "${elements[@]}"; 
        do
            ((COUNT++))
            OUTPUT_FILE="$OUTPUT_DIR/${DATASET}_output_strategy_${STRATEGY_ID}_case_${COUNT}_data_${DATASET}_repeat_${REPEAT}_society_${element}_agent_${AGENT}_turn_${TURN}.txt"
            EXP_TYPE="mixtral-agent-${AGENT}"
            nohup python run_agent.py --society $element --dataset $DATASET --turn $TURN --experiment_type $EXP_TYPE --agent $AGENT --model $MODEL --case_id "$CASE_ID" --api_idx $API_IDX --api_account $API_ACCOUNT > $OUTPUT_FILE 2>&1 &
        done
    done
done
