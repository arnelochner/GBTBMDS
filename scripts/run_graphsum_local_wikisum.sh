#!/usr/bin/env bash
set -eux

source ./env_local/env_local.sh
source ./env_local/utils.sh
source ./model_config/graphsum_model_conf_local_wikisum

if [ ! -d log  ];then
  mkdir log
else
  echo log exist
fi

if [ ! -d results  ];then
  mkdir results
else
  echo results exist
fi

if [ ! -d models  ];then
  mkdir models
else
  echo models exist
fi

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.98

# check
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP} \
                --selected_gpus 0,1,2,3,4,5,6,7 \
                --nproc_per_node 8"

python3.7 -u ./src/launch.py ${distributed_args} \
    ./src/run.py --model_name "graphsum" \
               --use_cuda true \
               --is_distributed false \
               --use_multi_gpu_test false \
               --use_fast_executor ${e_executor:-"true"} \
               --use_fp16 ${use_fp16:-"false"} \
               --use_dynamic_loss_scaling ${use_fp16} \
               --init_loss_scaling ${loss_scaling:-128} \
               --weight_sharing true \
               --do_train true \
               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size 10 \
               --in_tokens false \
               --stream_job ${STREAM_JOB:-""} \
               --init_pretraining_params ${MODEL_PATH:-""} \
               --train_set ${TASK_DATA_PATH}/train \
               --dev_set ${TASK_DATA_PATH}/valid \
               --test_set ${TASK_DATA_PATH}/test \
               --vocab_path ${VOCAB_PATH} \
               --config_path model_config/graphsum_config.json \
               --checkpoints ./models/graphsum_wikisum \
               --decode_path ./results/graphsum_wikisum \
               --lr_scheduler ${lr_scheduler} \
               --save_steps 100000 \
               --weight_decay ${WEIGHT_DECAY} \
               --warmup_steps ${WARMUP_STEPS} \
               --validation_steps 200000 \
               --epoch 100 \
               --max_para_num 30 \
               --max_para_len 150 \
               --max_tgt_len 400 \
               --max_out_len 400 \
               --min_out_len 20 \
               --graph_type "similarity" \
               --len_penalty 0.4 \
               --block_trigram True \
               --report_rouge True \
               --learning_rate ${LR_RATE} \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/graphsum_wikisum.log" \
               --random_seed 1 > log/lanch.log 2>&1
