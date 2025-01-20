#!/bin/bash
# 正样本不去冗余，负样本去冗余50%(8:1:1随机划分)
export CUDA_VISIBLE_DEVICES=0
# random seed
seed=1211

# for dataset
DATASET_NAME="extra_p_2_class_v3_case_01"
DATASET_TYPE="protein"
# for task
TASK_TYPE="binary_class"
TASK_LEVEL_TYPE="seq_level"
LABEL_TYPE="extra_p_2_class"

# for input
## seq, vector, matrix, seq_matrix, seq_vector
### sequence + embedding channels
INPUT_TYPE="seq_matrix"
## single or pair
INPUT_MODE="single"
TRUNC_TYPE="right"

# for model
MODEL_TYPE="lucaprot"
CONFIG_NAME="lucaprot_config.json"
FUSION_TYPE="concat"
dropout_prob=0.1
fc_size=256
classifier_size=$((fc_size + fc_size))
BEST_METRIC_TYPE="f1"
loss_type="bce"


## for sequence channel
SEQ_MAX_LENGTH=3072
hidden_size=1024
intermediate_size=4096
num_attention_heads=8
num_hidden_layers=4
### pooling type: none, max, mean, value_attention
SEQ_POOLING_TYPE="value_attention"
# char-level
seq_vocab_path="prot"

## for embedding channel
embedding_input_size=2560
matrix_max_length=3072
### pooling type: none, max, value_attention
MATRIX_POOLING_TYPE="value_attention"
### embedding llm
llm_version="esm2"
llm_type="esm"
llm_step="3B"

# for training
## max epochs
num_train_epochs=50
## accumulation gradient steps
gradient_accumulation_steps=2
# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数, 这里是4000）
logging_steps=4000
## checkpoint的间隔step数。-1表示按照epoch粒度保存checkpoint
save_steps=-1
## warmup_steps个step到达peak lr
warmup_steps=8000
## 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
## -1自动计算
max_steps=-1
## batch size for one GPU
batch_size=8
## 最大学习速率(peak learning rate)
learning_rate=2e-4
## data loading buffer size
buffer_size=10240
## positive weight
pos_weight=4.0

# model building time
time_str=$(date "+%Y%m%d%H%M%S")

cd ../..
python run.py \
  --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/train/ \
  --dev_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/dev/ \
  --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/test/ \
  --buffer_size $buffer_size \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --task_level_type $TASK_LEVEL_TYPE \
  --model_type $MODEL_TYPE \
  --input_type $INPUT_TYPE \
  --input_mode $INPUT_MODE \
  --label_type $LABEL_TYPE \
  --label_filepath ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
  --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --config_path ../config/$MODEL_TYPE/$CONFIG_NAME \
  --seq_vocab_path $seq_vocab_path \
  --seq_pooling_type $SEQ_POOLING_TYPE \
  --matrix_pooling_type $MATRIX_POOLING_TYPE \
  --fusion_type $FUSION_TYPE \
  --do_train \
  --do_eval \
  --do_predict \
  --do_metrics \
  --evaluate_during_training \
  --per_gpu_train_batch_size=$batch_size \
  --per_gpu_eval_batch_size=$batch_size  \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=$learning_rate \
  --lr_update_strategy step \
  --lr_decay_rate 0.95 \
  --num_train_epochs=$num_train_epochs \
  --overwrite_output_dir \
  --seed $seed \
  --sigmoid \
  --loss_type $loss_type \
  --best_metric_type $BEST_METRIC_TYPE \
  --seq_max_length=$SEQ_MAX_LENGTH \
  --embedding_input_size $embedding_input_size \
  --matrix_max_length=$matrix_max_length \
  --trunc_type=$TRUNC_TYPE \
  --pos_weight $pos_weight \
  --save_all \
  --llm_version $llm_version \
  --llm_type $llm_type \
  --llm_step $llm_step \
  --ignore_index -100 \
  --hidden_size $hidden_size \
  --intermediate_size $intermediate_size \
  --num_attention_heads $num_attention_heads \
  --num_hidden_layers $num_hidden_layers \
  --dropout_prob $dropout_prob \
  --classifier_size $classifier_size \
  --vector_dirpath /mnt/sanyuan.hy/workspace/vectors/lucapcycle/protein/esm/esm2/3B/ \
  --matrix_dirpath /mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B/#/mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
  --seq_fc_size $fc_size \
  --matrix_fc_size $fc_size \
  --vector_fc_size $fc_size \
  --emb_activate_func gelu \
  --fc_activate_func gelu \
  --classifier_activate_func gelu \
  --warmup_steps $warmup_steps \
  --beta1 0.9 \
  --beta2 0.99 \
  --weight_decay 0.01 \
  --save_steps $save_steps \
  --max_steps $max_steps \
  --logging_steps $logging_steps \
  --max_grad_norm 1.0 \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --matrix_embedding_exists \
  --matrix_add_special_token \
  --no_token_type_embeddings \
  --no_position_embeddings \
  --use_rotary_position_embeddings



