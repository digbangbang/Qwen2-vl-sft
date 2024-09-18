# bash file, debug on single machine
DS_CONFIG_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/examples/deepspeed/ds_z3_config.json'
OUTPUT_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/Qwen_sft_full'


deepspeed --include localhost:0 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/huggingface.co/Qwen/Qwen2-VL-7B-Instruct \
    --dataset_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data \
    --dataset viz_cls_2023_mini_sharegpt \
    --buffer_size 128 \
    --preprocessing_batch_size 128 \
    --streaming \
    --max_step 6740 \
    --dispatch_batches False \
    --template qwen2_vl \
    --finetuning_type lora \
    --flash_attn fa2 \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --ddp_timeout 9000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 1024 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 1 \
    --bf16 