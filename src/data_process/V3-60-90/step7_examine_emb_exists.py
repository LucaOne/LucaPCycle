#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/22 14:21
@project: LucaPCycleV3
@file: step7_examine_emb_exists
@desc: xxxx
"""
import os
import sys
import random
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import csv_reader, csv_writer, fasta_reader, write_fasta
    from utils import clean_special_char
except ImportError:
    from src.file_operator import csv_reader, csv_writer, fasta_reader, write_fasta
    from src.utils import clean_special_char
random.seed(1111)

prefix_path = "/mnt/sanyuan.hy/workspace/LucaPCycleV3/dataset"

dataset_name_list = [
    "extra_p_2_class_v3_case_01",
    "extra_p_2_class_v3_case_02",
    "extra_p_2_class_v3_case_03",
    "extra_p_2_class_v3_case_04",
    "extra_p_2_class_v3_case_05",
    "extra_p_2_class_v3_case_06",
    "extra_p_2_class_v3_case_07",
    "extra_p_2_class_v3_case_08",
    "extra_p_2_class_v3_case_09",
    "extra_p_2_class_v3_case_10",
    "extra_p_2_class_v3_case_11",
    "extra_p_2_class_v3_case_12"
]

middle_path = "protein/binary_class/"

emb_dir_list = [
    "/mnt/sanyuan.hy/workspace/matrices/lucapcycle_positives/protein/esm/esm2/3B",
    "/mnt/sanyuan.hy/workspace/matrices/lucapcycle_negatives/protein/esm/esm2/3B"
]


dataset_type_list = ["train", "dev", "test"]

for dataset_name in dataset_name_list:
    print("-" * 50)
    print("dataset: %s" % dataset_name)
    seq_id_set = set()
    seq_num = 0
    seq_emb_exists_num = 0
    seq_emb_exists_num_p = 0
    seq_emb_exists_num_n = 0
    seq_emb_not_exists_p = []
    seq_emb_not_exists_n = []
    for dataset_type in dataset_type_list:
        for row in csv_reader(os.path.join(prefix_path, dataset_name, middle_path, dataset_type, "%s.csv" % dataset_type)):
            seq_id = row[0]
            label = int(row[-1])
            seq_num += 1
            seq_id_set.add(seq_id)
            emb_filename = "matrix_%s.pt" % seq_id
            exists_flag = False
            for emb_dir in emb_dir_list:
                if os.path.exists(os.path.join(emb_dir, emb_filename)):
                    exists_flag = True
                    break
            if label == 0 and exists_flag:
                seq_emb_exists_num += 1
                seq_emb_exists_num_n += 1
            elif label == 0:
                seq_emb_not_exists_n.append([seq_id, row[2]])
            if label == 1 and exists_flag:
                seq_emb_exists_num += 1
                seq_emb_exists_num_p += 1
            elif label == 1:
                seq_emb_not_exists_p.append([seq_id, row[2]])
    print("seq_num: %d, seq_id_num: %d, seq_emb_exists_num: %d" % (seq_num, len(seq_id_set), seq_emb_exists_num))
    print("seq_emb_exists_num: %d, seq_emb_exists_num_p: %d, seq_emb_exists_num_n: %d" % (seq_emb_exists_num, seq_emb_exists_num_p, seq_emb_exists_num_n))
    write_fasta("./%s_seq_emb_not_exists_p.fasta" % dataset_name, sequences=seq_emb_not_exists_p)
    write_fasta("./%s_seq_emb_not_exists_n.fasta" % dataset_name, sequences=seq_emb_not_exists_n)
    print("-" * 50)
    