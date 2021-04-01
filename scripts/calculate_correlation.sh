#!/usr/bin/env bash

rouge_information_path=rouge_information/wikisum/
transformed_attention_weights_path=transformed_attention_weights/wikisum/
aggregation_metric="Mean"
aggregate_function="np.median"
result_output=correlation_results/wikisum/

python -u ./src/correlation_calculation/correlation_calculation.py \
                    --rouge_information_path $rouge_information_path\
                    --transformed_attention_weights_path $transformed_attention_weights_path\
                    --aggregation_metric $aggregation_metric\
                    --aggregate_function $aggregate_function\
                    --result_output $result_output