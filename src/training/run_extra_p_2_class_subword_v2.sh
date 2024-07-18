#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# stats
# positives: 214193, negative: 751331
# train: 955868
# dev: 4828
# test: 4828
DATASET_NAME="extra_p_2_class_v2"
DATASET_TYPE="protein"
TASK_TYPE="binary_class"
TASK_LEVEL_TYPE="seq_level"
MODEL_TYPE="lucaprot"
CONFIG_NAME="lucaprot_config.json"
# seq,vector,matrix,seq_matrix,seq_vector
INPUT_TYPE="seq_matrix"
INPUT_MODE="single"
LABEL_TYPE="extra_p_2_class_v2"
FUSION_TYPE="concat"
embedding_input_size=2560
SEQ_MAX_LENGTH=3072
matrix_max_length=3072
TRUNC_TYPE="right"
hidden_size=2560
num_attention_heads=8
num_hidden_layers=4
dropout_prob=0.1
# none, max, value_attention
SEQ_POOLING_TYPE="value_attention"
# none, max, value_attention
MATRIX_POOLING_TYPE="value_attention"
codes_file="extra_p_50_codes_20000.txt"
seq_subword="extra_p_50_subword_vocab_20000.txt"
BEST_METRIC_TYPE="f1"
classifier_size=1280
loss_type="bce"
llm_version="esm2"
llm_type="esm"
llm_step="3B"
batch_size=8
learning_rate=1e-4
gradient_accumulation_steps=8
time_str=$(date "+%Y%m%d%H%M%S")
cd ..
python run.py \
  --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/train/ \
  --dev_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/dev/ \
  --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/test/ \
  --buffer_size 10240 \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --task_level_type $TASK_LEVEL_TYPE \
  --model_type $MODEL_TYPE \
  --input_type $INPUT_TYPE \
  --input_mode $INPUT_MODE \
  --label_type $LABEL_TYPE \
  --seq_subword \
  --codes_file ../subword/$DATASET_NAME/$codes_file \
  --label_filepath ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
  --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --config_path ../config/$MODEL_TYPE/$CONFIG_NAME \
  --seq_vocab_path ../vocab/$DATASET_NAME/$seq_subword \
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
  --num_train_epochs=50 \
  --overwrite_output_dir \
  --seed 1221 \
  --sigmoid \
  --loss_type $loss_type \
  --best_metric_type $BEST_METRIC_TYPE \
  --seq_max_length=$SEQ_MAX_LENGTH \
  --embedding_input_size $embedding_input_size \
  --matrix_max_length=$matrix_max_length \
  --trunc_type=$TRUNC_TYPE \
  --no_token_embeddings \
  --no_token_type_embeddings \
  --no_position_embeddings \
  --pos_weight 4.0 \
  --save_all \
  --llm_version $llm_version \
  --llm_type $llm_type \
  --ignore_index -100 \
  --hidden_size $hidden_size \
  --num_attention_heads $num_attention_heads \
  --num_hidden_layers $num_hidden_layers \
  --dropout_prob $dropout_prob \
  --classifier_size $classifier_size \
  --vector_dirpath ../vectors/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$llm_version/$llm_type/$llm_step   \
  --matrix_dirpath ../matrices/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$llm_version/$llm_type/$llm_step  \
  --seq_fc_size null \
  --matrix_fc_size null \
  --vector_fc_size null \
  --emb_activate_func gelu \
  --fc_activate_func gelu \
  --classifier_activate_func gelu \
  --warmup_steps 8000 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight_decay 0.01 \
  --save_steps 1000000000 \
  --logging_steps 4000 \
  --not_append_eos \
  --not_prepend_bos



