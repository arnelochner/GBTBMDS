#!/usr/bin/env bash
set -eux




python -u ./src/data_preprocess/graphsum/preprocess_graphsum_data.py \
               -json_path "json_data/" \
               -data_path "MultiNews_data_tfidf_paddle" \
               -vocab_path "./data/spm9998_3.model" \
               -train_src "../multi-news-processed/train.txt.src" \
               -train_tgt "../multi-news-processed/train.txt.tgt" \
               -valid_src "../multi-news-processed/val.txt.src" \
               -valid_tgt "../multi-news-processed/val.txt.tgt" \
               -test_src "../multi-news-processed/test.txt.src" \
               -test_tgt "../multi-news-processed/test.txt.tgt" \
               -max_nsents 30 \
               -sentence_level True\
               -sim_function "tf-idf" > log/preprocess.log 2>&1
