# bash file, debug for lora

# export CUDA_VISIBLE_DEVICES=1

# DISTRIBUTED_ARGS="
#     --nproc_per_node 1 \
#     --nnodes 1 \
#     --node_rank 0 \
#     --master_addr localhost \
#     --master_port 29500
#     "

# DS_CONFIG_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/examples/deepspeed/ds_z3_config.json'
# OUTPUT_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/Qwen_out'
# torchrun $DISTRIBUTED_ARGS /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/src/train.py \
#     --deepspeed $DS_CONFIG_PATH \
#     --stage sft \
#     --do_train \
#     --model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/huggingface.co/Qwen/Qwen2-VL-7B-Instruct \
#     --dataset_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data \
#     --dataset viz_cls_2023_mini_sharegpt \
#     --template qwen2_vl \
#     --preprocessing_num_workers 32 \
#     --finetuning_type lora \
#     --flash_attn fa2 \
#     --output_dir $OUTPUT_PATH \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --warmup_steps 100 \
#     --weight_decay 0.1 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --ddp_timeout 9000 \
#     --learning_rate 5e-6 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --cutoff_len 1024 \
#     --save_steps 1000 \
#     --plot_loss \
#     --num_train_epochs 1 \
#     --bf16 



DS_CONFIG_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/examples/deepspeed/ds_z3_config.json'
OUTPUT_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/Qwen_out'
deepspeed --include localhost:1 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/huggingface.co/Qwen/Qwen2-VL-7B-Instruct \
    --dataset_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data \
    --dataset viz_cls_2023_mini_sharegpt \
    --template qwen2_vl \
    --preprocessing_num_workers 32 \
    --finetuning_type lora \
    --flash_attn fa2 \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 1024 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 1 \
    --bf16 