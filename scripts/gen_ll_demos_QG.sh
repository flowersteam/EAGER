#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/make_agent_demos_QG.py \
--env $1 \
--episodes 7500 --valid-episodes 1000 \
--include-goal --QG-generation \
--gen-no-answer-question True \
--seed 1
# --gen-no-answer-question True
# --biased True
#--episodes 15000