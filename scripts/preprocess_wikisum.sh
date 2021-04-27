#!/usr/bin/env bash
set -eux




python -u ./src/data_preprocess/graphsum/preprocess_graphsum_data.py \
               -mds_dataset "WikiSum/" \
               -ranked_wikisum_data_path "data/WikiSum/test" \
               -add_dummy_number_of_textual_units true \
               -num_examples 10\
               -wikisum_output_path "data/WikiSum/10_test" > log/preprocess.log 2>&1
