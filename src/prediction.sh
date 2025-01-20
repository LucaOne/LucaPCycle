# Binary Classification
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python prediction.py \
    --seq_type prot \
    --input_file ../test_data/examples.fasta \
    --llm_truncation_seq_length 10240 \
    --model_path .. \
    --save_path ../predicted_results/test_data/examples_predicted.csv \
    --dataset_name extra_p_2_class_v3 \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucaprot \
    --input_type seq_matrix \
    --input_mode single \
    --time_str 20240924203640 \
    --step 264284 \
    --threshold 0.2 \
    --per_num 10000 \
    --gpu_id 0

# Multi-Class Classification
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python prediction.py \
    --seq_type prot \
    --input_file ../test_data/example_positives.fasta \
    --llm_truncation_seq_length 10240 \
    --model_path .. \
    --save_path ../predicted_results/test_data/example_positives_fine_grained_predicted.csv \
    --dataset_name extra_p_31_class_v3 \
    --dataset_type protein \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type lucaprot \
    --input_type seq_matrix \
    --input_mode single \
    --time_str 20240923094428 \
    --step 8569250 \
    --topk $topk \
    --per_num 10000 \
    --gpu_id 0

