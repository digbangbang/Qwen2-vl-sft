# bash file for sft on the viz data with multi node multi gpu, given the nodes and gpus number in the hope file
num_nodes=$1
num_gpus=$2
echo "num nodes is $num_nodes"
echo "num gpus is $num_gpus"

cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"                                                                                                                                
worker_list_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/liulei82/envs/env_config/conda_base
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/envs/lzw_qwen

export PATH=$PATH:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/projs/food_ocr/third_party/pdsh-2.34/bin
mkdir ~/.ssh/
cp -r /mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/dev/misc/ssh/* ~/.ssh/
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

flag_filedir="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/mycode/job_finished_flag"
flag_filepath=$flag_filedir$master
echo "flag filepath is $flag_filepath"

if (($node_rank == 0))
then
    generate_hostfile_command="import generate_hostfile;print(generate_hostfile.parse(\"$cluster_spec\", \"worker\"))"
    eval hostfile_path=`python -c "$generate_hostfile_command"`
    echo "##########hostfile###############"
    echo "hostfile_path is $hostfile_path"
    cat $hostfile_path
fi

if (($node_rank == 0))
then
    wait_cnt=0
    while ((wait_cnt <= 720))
    do
        all_worker_ready=true
        for (( i=1; i<${#worker_strs[@]}; i++ ));
        do
            if [ ! -f "$flag_filedir${worker_strs[$i]}" ];
            then
                echo "worker $i is not ready!"
                all_worker_ready=false
                break
            fi
        done
        if [ $all_worker_ready=true ];
        then
            echo "All workers have been ready! start train ..."
            break
        else
            echo "Waiting other workers to be ready ..."
            sleep 5
        fi
        ((wait_cnt++))
    done
fi


DS_CONFIG_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/examples/deepspeed/ds_z3_config.json'
OUTPUT_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/Qwen_sft_full_mix'

if (($node_rank == 0))
then
    deepspeed --num_nodes $num_nodes --num_gpus $num_gpus --hostfile=$hostfile_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/src/train.py \
        --deepspeed $DS_CONFIG_PATH \
        --stage sft \
        --do_train \
        --model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/huggingface.co/Qwen/Qwen2-VL-7B-Instruct \
        --dataset_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data \
        --template qwen2_vl \
        --finetuning_type full \
        --dataset viz_1,viz_2,viz_3,viz_4,viz_5,viz_6,viz_7,viz_8,viz_9,viz_10 \
        --buffer_size 16384 \
        --preprocessing_batch_size 256 \
        --preprocessing_num_workers 64 \
        --mix_strategy interleave_under \
        --interleave_probs 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 \
        --streaming \
        --max_step 3370 \
        --dispatch_batches False \
        --flash_attn fa2 \
        --output_dir $OUTPUT_PATH \
        --overwrite_cache \
        --overwrite_output_dir \
        --warmup_steps 200 \
        --weight_decay 0.1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --ddp_timeout 9000 \
        --learning_rate 8e-6 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --cutoff_len 2048 \
        --save_steps 500 \
        --plot_loss \
        --num_train_epochs 1 \
        --bf16 
    touch $flag_filepath
    echo "master job finished, generate flag file $flag_filepath"
else
   node_ready_flag_filepath=$flag_filedir${worker_strs[$node_rank]}
   sleep 20
   touch $node_ready_flag_filepath
   echo "node $node_rank is ready, generate ready flag file $node_ready_flag_filepath"
   while [ ! -f $flag_filepath ]
   do
      echo "master job doesnot finished, continue sleep..."
      sleep 30
   done
fi