#!/bin/bash

DATASET="chess"
AGENT=3
TURN=3
MODEL="gpt-3.5-turbo-1106"
EXP_TYPE="gpt-1106-main"
STRATEGIES="['000','001','010','011','100','101','110','111']"
API_IDX=0
API_ACCOUNT="openai"

OUTPUT_DIR="log-main"
mkdir -p $OUTPUT_DIR

for ROLE in {0..3}
do
    for REPEAT in {1..5}
    do
        mkdir -p "results/$EXP_TYPE/$DATASET/$REPEAT"
        COUNT=0
        for CASE_ID in "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"
        do
            ((COUNT++))
            OUTPUT_FILE="$OUTPUT_DIR/${DATASET}_output_strategy_${STRATEGY_ID}_case_${COUNT}_data_${DATASET}_repeat_${REPEAT}_role_${ROLE}_agent_${AGENT}_turn_${TURN}.txt"
            nohup run_main.py --role $ROLE --dataset $DATASET --repeat $REPEAT --turn $TURN --api_idx $API_IDX --api_account $API_ACCOUNT --experiment_type $EXP_TYPE --model $MODEL --agent $AGENT --strategies "$STRATEGIES" --case_id "$CASE_ID"  > $OUTPUT_FILE 2>&1 &
         done
        # ((API_IDX++))
    done
done
