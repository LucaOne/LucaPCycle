#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/16 22:17
@project: LucaPCycle
@file: data_verfiy
@desc: xxxx
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
except ImportError:
    from src.file_operator import csv_reader, csv_writer, fasta_reader, write_fasta

positive_seqs = set()

negative_seqs = set()

filepath_list = [
    "../../../dataset/extra_p_2_class_v2/protein/binary_class/train/train.csv",
    "../../../dataset/extra_p_2_class_v2/protein/binary_class/dev/dev.csv",
    "../../../dataset/extra_p_2_class_v2/protein/binary_class/test/test.csv"
    ]
pos_num = 0
neg_num = 0
for filepath in filepath_list:
    for row in csv_reader(filepath):
        seq = row[2].upper().strip()
        label = int(row[-1])
        if label == 1:
            positive_seqs.add(seq)
            pos_num += 1
        else:
            negative_seqs.add(seq)
            neg_num += 1

print("%d, %d" % (len(positive_seqs), pos_num))
print("%d, %d" % (len(negative_seqs), neg_num))
print(len(positive_seqs.intersection(negative_seqs)))
