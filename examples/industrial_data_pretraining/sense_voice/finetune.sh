# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`
stage=1
stop_stage=3
# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically

#model_name_or_model_dir="iic/SenseVoiceSmall"

model_name_or_model_dir='/data/guyf/.cache/modelscope/hub/iic/SenseVoiceSmall'

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}
wavscp_dir='/data/guyf/funasr/FunASR_sv/data/list'
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: 生成wav.scp"
  python list_test.py \
  --tra_file='/data/guyf/funasr/FunASR/examples/aishell/raw_data/data_aishell/wav/train' \
  --output_dir=${wavscp_dir} \
  --txt_tra='/data/guyf/funasr/FunASR_sv/data/list/text_tra.txt' \
  --dev_file='/data/guyf/funasr/FunASR/examples/aishell/raw_data/data_aishell/wav/dev' \
  --txt_dev='/data/guyf/funasr/FunASR_sv/data/list/text_dev.txt'
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: 生成json"
  python ../../../funasr/datasets/audio_datasets/sensevoice2jsonl.py \
  ++scp_file_list='["/data/guyf/funasr/FunASR_sv/data/list/wav_tra.scp", "/data/guyf/funasr/FunASR_sv/data/list/text_tra.txt"]' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="/data/guyf/funasr/FunASR_sv/data/tra_example.jsonl" \
  ++model_dir='iic/SenseVoiceSmall'

  python ../../../funasr/datasets/audio_datasets/sensevoice2jsonl.py \
  ++scp_file_list='["/data/guyf/funasr/FunASR_sv/data/list/wav_dev.scp", "/data/guyf/funasr/FunASR_sv/data/list/text_dev.txt"]' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="/data/guyf/funasr/FunASR_sv/data/val_example.jsonl" \
  ++model_dir='iic/SenseVoiceSmall'
fi
# data dir, which contains: train.json, val.json
train_data=${workspace}/data/train_example.jsonl
val_data=${workspace}/data/val_example.jsonl
# exp output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/../../ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"
echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool=/data/guyf/funasr/FunASR_sv/funasr/bin/train_ds.py    #!!修改路径
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: 微调"
  torchrun $DISTRIBUTED_ARGS \
  ${train_tool} \
  ++model="${model_name_or_model_dir}" \
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++dataset_conf.data_split_num=1 \
  ++dataset_conf.batch_sampler="BatchSampler" \
  ++dataset_conf.batch_size=6000  \
  ++dataset_conf.sort_size=1024 \
  ++dataset_conf.batch_type="token" \
  ++dataset_conf.num_workers=4 \
  ++train_conf.max_epoch=50 \
  ++train_conf.log_interval=1 \
  ++train_conf.resume=true \
  ++train_conf.validate_interval=2000 \
  ++train_conf.save_checkpoint_interval=2000 \
  ++train_conf.keep_nbest_models=20 \
  ++train_conf.avg_nbest_model=10 \
  ++train_conf.use_deepspeed=false \
  ++train_conf.deepspeed_config=${deepspeed_config} \
  ++optim_conf.lr=0.0002 \
  ++output_dir="${output_dir}" &> ${log_file}
fi