#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
python babyai/scripts/result_QA_study.py