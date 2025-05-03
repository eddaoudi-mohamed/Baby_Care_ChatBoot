PRE_SEQ_LEN=128
LR=2e-2
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file data/dev.json \
    --validation_file data/train.json \
    --prompt_column content \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path chatglm-6b-int4 \
    --output_dir output/BabyCarChat\
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

