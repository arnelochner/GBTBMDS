set FLAGS_eager_delete_tensor_gb=1.0
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.98
set PADDLE_IS_LOCAL=1

set graphsum = "graphsum"
set use_cuda = "true"
set is_distributed = "false"
set use_multi_gpu_test= "true"
set use_fast_executor= "true"
set use_fp16 = "false"
set use_dynamic_loss_scaling = "false"
set init_loss_scaling = -128 
set weight_sharing = "true"
set do_train = "true"
set do_val = "false"
set do_test = "false"
set do_dec = "true"
set verbose = "true"
set batch_size = 512
set in_tokens = "true"
set stream_job = ""
set init_pretraining_params = ""
set train_set = "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/small_train"
set dev_set = "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/valid"
set test_set = "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/test"
set vocab_path = "E:/graphsum/data/spm9998_3.model"
set config_path = "./model_config/graphsum_config.json"
set checkpoints = "./models/graphsum_multinews"
set decode_path = "./results/graphsum_multinews"
set lr_scheduler = "noam_decay"
set save_steps = 10000
set weight_decay = 0.01
set warmup_steps= 8000
set validation_steps = 20000
set 

               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size 512 \
               --in_tokens true \
               --stream_job "" \
               --init_pretraining_params "" \
               --train_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/small_train" \
               --dev_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/valid" \
               --test_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/test" \
               --vocab_path "E:/graphsum/data/spm9998_3.model" \
               --config_path ./model_config/graphsum_config.json \
               --checkpoints ./models/graphsum_multinews \
               --decode_path ./results/graphsum_multinews \
               --lr_scheduler "noam_decay" \
               --save_steps 10000 \
               --weight_decay 0.01 \
               --warmup_steps 8000 \
               --validation_steps 20000 \
               --epoch 100 \
               --max_para_num 30 \
               --max_para_len 60 \
               --max_tgt_len 300 \
               --max_out_len 300 \
               --min_out_len 200 \
               --graph_type "similarity" \
               --len_penalty 0.6 \
               --block_trigram True \
               --report_rouge True \
               --learning_rate 2.0 \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/graphsum_multinews.log" \
               --random_seed 1

python "src/run.py"  --model_name "graphsum" \
               --use_cuda true \
               --is_distributed false \
               --use_multi_gpu_test true \
               --use_fast_executor true \
               --use_fp16 false \
               --use_dynamic_loss_scaling false \
               --init_loss_scaling -128 \
               --weight_sharing true \
               --do_train true \
               --do_val false \
               --do_test true \
               --do_dec true \
               --verbose true \
               --batch_size 512 \
               --in_tokens true \
               --stream_job "" \
               --init_pretraining_params "" \
               --train_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/small_train" \
               --dev_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/valid" \
               --test_set "E:/graphsum/data/MultiNews_data_tfidf_30_paddle/test" \
               --vocab_path "E:/graphsum/data/spm9998_3.model" \
               --config_path ./model_config/graphsum_config.json \
               --checkpoints ./models/graphsum_multinews \
               --decode_path ./results/graphsum_multinews \
               --lr_scheduler "noam_decay" \
               --save_steps 10000 \
               --weight_decay 0.01 \
               --warmup_steps 8000 \
               --validation_steps 20000 \
               --epoch 100 \
               --max_para_num 30 \
               --max_para_len 60 \
               --max_tgt_len 300 \
               --max_out_len 300 \
               --min_out_len 200 \
               --graph_type "similarity" \
               --len_penalty 0.6 \
               --block_trigram True \
               --report_rouge True \
               --learning_rate 2.0 \
               --skip_steps 100 \
               --grad_norm 2.0 \
               --pos_win 2.0 \
               --label_smooth_eps 0.1 \
               --num_iteration_per_drop_scope 10 \
               --log_file "log/graphsum_multinews.log" \
               --random_seed 1