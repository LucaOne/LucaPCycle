#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/16 22:17
@project: LucaPCycleV3
@file: step8_dataset_verify
@desc: dataset verify
"""
import os
import sys
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


for idx in range(1, 5):
    print("case: %d" % idx)
    positive_seqs = set()
    negative_seqs = set()
    positive_seq_ids = set()
    negative_seq_ids = set()
    filepath_list = [
        "../../../dataset/extra_p_2_class_v3_case_%02d/protein/binary_class/train/train.csv" % idx,
        "../../../dataset/extra_p_2_class_v3_case_%02d/protein/binary_class/dev/dev.csv" % idx,
        "../../../dataset/extra_p_2_class_v3_case_%02d/protein/binary_class/test/test.csv" % idx
    ]
    pos_num = 0
    neg_num = 0
    seq_id_2_seq = {}
    for filepath in filepath_list:
        for row in csv_reader(filepath):
            seq_id = row[0]
            seq = clean_special_char(seq_id, row[2])

            label = int(row[-1])
            if label == 1:
                if seq_id in positive_seq_ids:
                    assert seq_id_2_seq[seq_id] == seq
                else:
                    positive_seq_ids.add(seq_id)
                    seq_id_2_seq[seq_id] = seq
                positive_seqs.add(seq)
                pos_num += 1
            else:
                if seq_id in negative_seq_ids:
                    assert seq_id_2_seq[seq_id] == seq
                else:
                    negative_seq_ids.add(seq_id)
                    seq_id_2_seq[seq_id] = seq
                negative_seqs.add(seq)
                neg_num += 1

    print("%d, %d, %d" % (len(positive_seq_ids), len(positive_seqs), pos_num))
    print("%d, %d, %d" % (len(negative_seq_ids), len(negative_seqs), neg_num))
    print(len(positive_seq_ids.intersection(negative_seq_ids)))
    print(len(positive_seqs.intersection(negative_seqs)))
    print("*" * 50)
