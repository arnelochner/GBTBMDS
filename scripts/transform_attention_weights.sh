#!/usr/bin/env bash

input_path=saved_attention_weights/
output_path=transformed_attention_weights/


python -u ./src/transformation/transform_attention_weights.py \
                    --input_path $input_path\
                    --output_path $output_path\