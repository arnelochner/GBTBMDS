#!/usr/bin/env bash
set -eux


if [ ! -d ../preprocess_data  ];then
  mkdir results
else
  echo results exist
fi



distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3,4,5,6,7 \
                --nproc_per_node 8"

python3.7 -u ./src/data_preprocess/preprocess_graphsum_data.py \
               --json_path "../preprocess_data/json_data" \
               --data_path "../preprocess_data/MultiNews_data_tfidf_paddle" \
               --vocab_path "./data/spm9998_3.model" \
               --train_src "../multi-news-processed/train.txt.src" \
               --train_tgt "../multi-news-processed/train.txt.tgt" \
               --valid_src "../multi-news-processed/valid.txt.src" \
               --valid_tgt "../multi-news-processed/train.txt.tgt" \
               --test_src "../multi-news-processed/test.txt.src" \
               --test_tgt "../multi-news-processed/train.txt.tgt" \
               --num_examples 1 \
               --sentence_level False\
               --sim_function "tf-idf" > log/preprocess.log 2>&1
