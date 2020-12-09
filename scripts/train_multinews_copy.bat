set FLAGS_eager_delete_tensor_gb=1.0
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.98
set PADDLE_IS_LOCAL=1

python "src/run.py"  --model_name "graphsum" ^
    --use_cuda true ^
    --is_distributed false ^
    --use_multi_gpu_test true ^
    --use_fast_executor true ^
    --use_fp16 false ^
    --use_dynamic_loss_scaling false ^
    --init_loss_scaling -128 ^
    --weight_sharing true ^
    --do_train true ^
    --do_val false ^
    --do_test false ^
    --do_dec true ^
    --verbose true ^
    --batch_size 512 ^
    --in_tokens true ^
    --train_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/small_train" ^
    --dev_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/valid" ^
    --test_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/test" ^
    --vocab_path "E:/graphsum/data/spm9998_3.model" ^
    --config_path ./model_config/graphsum_config.json ^
    --checkpoints ./models/graphsum_multinews ^
    --decode_path ./results/graphsum_multinews ^
    --lr_scheduler "noam_decay" ^
    --save_steps 10000 ^
    --weight_decay 0.01 ^
    --warmup_steps 8000 ^
    --validation_steps 20000 ^
    --epoch 100 ^
    --max_para_num 30 ^
    --max_para_len 60 ^
    --max_tgt_len 300 ^
    --max_out_len 300 ^
    --min_out_len 200 ^
    --graph_type "similarity" ^
    --len_penalty 0.6 ^
    --block_trigram True ^
    --report_rouge True ^
    --learning_rate 2.0 ^
    --skip_steps 100 ^
    --grad_norm 2.0 ^
    --pos_win 2.0 ^
    --label_smooth_eps 0.1 ^
    --num_iteration_per_drop_scope 10 ^
    --log_file "log/graphsum_multinews.log" ^
    --random_seed 1