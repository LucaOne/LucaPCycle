# ESM embedding inference for positives
cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_positives_part_01_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_positives_part_02_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1

cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_positives_part_03_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2

cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_positives_part_04_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3


# ESM embedding inference for negatives
cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_01_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_02_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1

cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_03_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2

cd ../src/llm/esm/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_04_04.fasta \
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_02_04_part_01_03.fasta\
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_02_04_part_02_03.fasta\
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm \
    --llm_version esm2 \
    --llm_step 3B \
    --embedding_type matrix \
    --trunc_type right \
    --truncation_seq_length 4094 \
    --input_file ../../../data/fasta/seq_id_rename_negatives_part_02_04_part_03_03.fasta\
    --seq_type prot \
    --save_path  /mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B/ \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3