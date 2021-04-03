#!/usr/bin/env bash
set -eux

source ./env_local/env_local.sh
source ./env_local/utils.sh
source ./model_config/graphsum_model_conf_local_multinews_paragraphs

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.98

attention_weights_path=saved_attention_weights/multinews

decode_path=results/graphsum_multinews

if [ ! -d log/multinews  ];then
  mkdir log/multinews
else
  echo log/multinews exist
fi

if [ ! -d $decode_path  ];then
  mkdir $decode_path
else
  echo $decode_path exist
fi


if [ ! -d $attention_weights_path  ];then
  mkdir $attention_weights_path
else
  echo $attention_weights_path exist
fi


python -u ./src/run.py \
               --model_name "graphsum" \
               --use_cuda true \
               --is_distributed false \
               --use_multi_gpu_test False \
               --use_fast_executor ${e_executor:-"true"} \
               --use_fp16 ${use_fp16:-"false"} \
               --use_dynamic_loss_scaling ${use_fp16} \
               --init_loss_scaling ${loss_scaling:-128} \
               --weight_sharing true \
               --do_train false \
               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size 12000 \
               --in_tokens true \
               --stream_job ${STREAM_JOB:-""} \
               --init_pretraining_params ${MODEL_PATH:-""} \
               --train_set ${TASK_DATA_PATH}/train \
               --dev_set ${TASK_DATA_PATH}/valid \
               --test_set ${TASK_DATA_PATH}/small_test \
               --vocab_path ${VOCAB_PATH} \
               --config_path model_config/graphsum_config.json \
               --checkpoints ./models/graphsum_multinews \
               --init_checkpoint ./models/multinews_downloaded_model/step_42976 \
               --attention_weights_path $attention_weights_path \
               --decode_path ./results/graphsum_multinews \
               --lr_scheduler ${lr_scheduler} \
               --save_steps 10000 \
               --weight_decay ${WEIGHT_DECAY} \
               --warmup_steps ${WARMUP_STEPS} \
               --validation_steps 20000 \
               --epoch 100 \
               --max_para_num 30 \
               --max_para_len 60 \
               --max_tgt_len 300 \
               --max_out_len 300 \
               --min_out_len 200 \
               --beam_size 5 \
               --graph_type "similarity" \
               --len_penalty 0.6 \
               --block_trigram True \
               --report_rouge True \
               --learning_rate ${LR_RATE} \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/multinews/rq2.log" \
               --random_seed 1 > log/multinews/launch_rq2.log 2>&1

               
               
echo "GraphSum predictions are done"

transformed_attention_weights_path=transformed_attention_weights/multinews/


python -u ./src/transformation/transform_attention_weights.py \
                    --input_path $attention_weights_path\
                    --output_path $transformed_attention_weights_path\
                    --max_beam_length 300 \
                    --only_highest_beam True

echo "Transformation of Global Attention Weights is done"

can_path=${decode_path}/test_final_preds.candidate
input_data=${TASK_DATA_PATH}/small_test
rouge_information_path=rouge_information/multinews/


python -u ./src/rouge_calculation/rouge.py \
                    --can_path $can_path\
                    --input_data $input_data\
                    --output_dir $rouge_information_path\
                    --spm_path ${VOCAB_PATH} > /dev/null 2>&1

echo "Rouge Calculation is done!"
                    
aggregation_metric="Mean"
aggregate_function="np.mean"
result_output=correlation_results/multinews/

python -u ./src/correlation_calculation/correlation_calculation.py \
                    --rouge_information_path $rouge_information_path\
                    --transformed_attention_weights_path $transformed_attention_weights_path\
                    --aggregation_metric $aggregation_metric\
                    --aggregate_function $aggregate_function\
                    --result_output $result_output
                
echo "Correlation Calculation is done!"