#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/18 22:54
@project: LucaPCycleV3
@file: step2_negatives
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
    from utils import clean_special_char
except ImportError:
    from src.file_operator import csv_reader, csv_writer, fasta_reader, write_fasta
    from src.utils import clean_special_char

assert os.path.exists("../../../data/fasta/seq_id_rename_positives.fasta")
assert os.path.exists("../../../data/fasta/seq_id_rename.csv")
assert not os.path.exists("../../../data/fasta/seq_id_rename_negatives.fasta")

positive_filepath = "../../../data/fasta/seq_id_rename_positives.fasta"
positives = {}
for row in fasta_reader(positive_filepath):
    seq_id, seq = row[0], clean_special_char(row[0], row[1])
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    positives[seq_id] = seq


data_filepath = "../../../data/fasta/seq_id_rename.csv"
rename_negatives = []
negative_min_seq_idx = None
contain_positive_num = 0
for row in csv_reader(data_filepath, header=True, header_filter=True):
    seq_id, ori_seq_id, seq = row[0], row[1], clean_special_char(row[1], row[2])
    if seq_id in positives:
        assert positives[seq_id] == seq
        contain_positive_num += 1
    else:
        rename_negatives.append([seq_id, seq])
        seq_idx = int(seq_id.replace("seq_", ""))
        if negative_min_seq_idx is None or negative_min_seq_idx > seq_idx:
            negative_min_seq_idx = seq_idx
print("contain_positive_num: %d" % contain_positive_num)
print("negative_min_seq_idx: %d" % negative_min_seq_idx)
print("rename_negatives: %d" % len(rename_negatives))

write_fasta(filepath="../../../data/fasta/seq_id_rename_negatives.fasta", sequences=rename_negatives)
rename_negative_seq_ids = set([item[0] for item in rename_negatives])
rename_negative_seqs = set([item[1] for item in rename_negatives])
print("rename_negatives: %d, rename_negative_seq_ids: %d, rename_negative_seqs: %d" % (len(rename_negatives), len(rename_negative_seq_ids), len(rename_negative_seqs)))
rename_negative_seq_lens = [len(seq) for seq in rename_negative_seqs]
print("negative seq len: 25%%: %d, 40%%: %d, 50%%: %d, 60%%: %d, 75%%: %d, 80%%: %d, 85%%: %d, 90%%: %d, 95%%: %d, 99%%: %d, median: %d, mean: %f, max: %d, min: %d" % (
    np.percentile(rename_negative_seq_lens, 25),
    np.percentile(rename_negative_seq_lens, 40),
    np.percentile(rename_negative_seq_lens, 50),
    np.percentile(rename_negative_seq_lens, 60),
    np.percentile(rename_negative_seq_lens, 75),
    np.percentile(rename_negative_seq_lens, 80),
    np.percentile(rename_negative_seq_lens, 85),
    np.percentile(rename_negative_seq_lens, 90),
    np.percentile(rename_negative_seq_lens, 95),
    np.percentile(rename_negative_seq_lens, 99),
    np.median(rename_negative_seq_lens),
    np.mean(rename_negative_seq_lens),
    np.max(rename_negative_seq_lens),
    np.min(rename_negative_seq_lens)
))

"""
contain_positive_num: 214193
negative_min_seq_idx: 214194
rename_negatives: 3822247
rename_negatives: 3822247, rename_negative_seq_ids: 3822247, rename_negative_seqs: 3812830
negative seq len: 25%: 206, 40%: 278, 50%: 331, 60%: 390, 75%: 494, 80%: 535, 85%: 608, 90%: 743, 95%: 1023, 99%: 1901, median: 331, mean: 417.062424, max: 35213, min: 2
"""