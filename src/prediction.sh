# Binary Classification
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python prediction.py \
    --fasta ../test_data/examples.fasta \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicted_results/test_data/examples_predicted.csv \
    --dataset_name extra_p_2_class_v2 \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucaprot \
    --input_type seq_matrix \
    --time_str 20240120061735 \
    --step 955872 \
    --threshold 0.2 \
    --per_num 1000 \
    --gpu_id 0

# Multi Classification
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python prediction.py \
    --fasta ../test_data/example_positives.fasta \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicted_results/test_data/example_positives_fine_grained_predicted.csv \
    --dataset_name extra_p_31_class_v2 \
    --dataset_type protein \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type lucaprot \
    --input_type seq_matrix \
    --time_str 20240120061524 \
    --step 294536 \
    --per_num 1000 \
    --gpu_id 1

