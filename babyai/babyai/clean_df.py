import os
import sys
import re
import numpy as np
from matplotlib import pyplot
import pandas


answer = input("Continue?")
if answer.lower() in ["y", "yes"]:

    list_to_clean = ['storage/logs/QG_QA/paral_SEQ-adjusted-train_env-multienv2_no_answer-lambda_026-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNM-adjusted-train_env-multienv2_no_answer-lambda_16-model-0_7-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLA-adjusted-train_env-multienv2_no_answer-lambda_043-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLA-adjusted-train_env-multienv2_no_answer-lambda_043-model-0_7-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNM-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_2/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-0-lambda_24-model-0_10-seed_1/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-0-lambda_24-model-0_10-seed_3/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_2/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_3/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-0-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_4/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_2/log.csv',
                     'storage/logs/QG_QA/PNL-adjusted_biased-1-train_env-PNL_no_answer_biased_debiased_QA-1-lambda_48-model-0_7-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_POP-adjusted-train_env-multienv2_no_answer-lambda_28-model-0_7-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_POP-adjusted-train_env-multienv2_no_answer-lambda_28-model-0_7-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_POP-adjusted-train_env-multienv2_no_answer-lambda_28-model-0_7-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLsa-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLsa-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLsa-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLsa-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNMsa-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNMsa-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNMsa-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNMsa-adjusted-train_env-multienv3_no_answer-lambda_16-model-0_10-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_POP-adjusted-train_env-multienv3_no_answer-lambda_38-model-0_10-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_0-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_2-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_3-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_9-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_1/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_3/log.csv',
                     'storage/logs/QG_QA/paral_PNLpar-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_4/log.csv',
                     'storage/logs/QG_QA/paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_24-model-0_10-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_48-model-0_10-seed_2/log.csv',
                     'storage/logs/QG_QA/paral_UNLM-adjusted-train_env-multienv3_no_answer-lambda_48-model-0_10-seed_3/log.csv']
    for a in list_to_clean:
        print(a)
        df = pandas.read_csv(a)
        print(df.head())
        while df.iloc[0]['frames'] == 'frames':
            df = df.drop([0])
            print(df.head())
            df.to_csv(a, index=False)
            df = pandas.read_csv(a)
        print("before saving")
        print(df.head())
        df.to_csv(a, index=False)

elif answer.lower() in ["n", "no"]:
    sys.exit("")
else:
    sys.exit("wrong command")



