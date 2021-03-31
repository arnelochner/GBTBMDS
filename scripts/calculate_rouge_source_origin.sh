#!/usr/bin/env bash

can_path=results/graphsum_multinews/test_final_preds.candidate
input_data=data/MultiNews_data_tfidf_30_paddle_full_paragraph/small_test/
output_dir=rouge_information/


python -u ./src/rouge_calculation/rouge.py \
                    --can_path $can_path\
                    --input_data $input_data\
                    --output_dir $output_dir