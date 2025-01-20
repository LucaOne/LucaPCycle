#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/16 22:17
@project: LucaPCycleV3
@file: step1_data_verfiy
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

assert not os.path.exists("../../../data/fasta/seq_id_rename.csv")
assert not os.path.exists("../../../data/fasta/seq_id_rename_positives.fasta")

dirpath_list = [
    "../../../data/fasta/positives",
    "../../../data/fasta/negatives"
]

sample_num = 0
positive_sample_num = 0
negative_sample_num = 0
seq_ids_set = set()
positive_seq_ids_set = set()
negative_seq_ids_set = set()
seqs_set = set()
positive_seqs_set = set()
negative_seqs_set = set()
seq_id_2_seq = {}
seq_2_seq_id = {}

seq_id_rename = []
seq_id_rename_idx = 1
rename_seq_ids = set()
rename_positives = []

for dirpath in dirpath_list:
    for filename in os.listdir(dirpath):
        if not filename.endswith(".fasta"):
            continue
        if "negative" in dirpath:
            dtype = filename.replace("lucapcycle_cold_spring_ori_negatives_", "").replace("_fasta.fasta", "")
        else:
            dtype = "positive"
        for row in fasta_reader(os.path.join(dirpath, filename)):
            seq_id = row[0]
            seq = clean_special_char(seq_id, row[1])
            if seq_id[0] == ">":
                seq_id = seq_id[1:]
            seq_ids_set.add(seq_id)
            seqs_set.add(seq)
            sample_num += 1
            if seq_id in seq_id_2_seq:
                if seq_id_2_seq[seq_id][0] != seq:
                    print(seq_id, dtype, ":", seq)
                    print(seq_id, seq_id_2_seq[seq_id][1], ":", seq_id_2_seq[seq_id][0])
                    print("*" * 50)
                    # id相同，但序列不同，重命名id
                    seq_id_rename.append(["seq_%d" % seq_id_rename_idx, seq_id, seq])
                    rename_seq_ids.add("seq_%d" % seq_id_rename_idx)
                    if dtype == "positive":
                        rename_positives.append(["seq_%d" % seq_id_rename_idx, seq])
                    seq_id_rename_idx += 1
                else:
                    # 相同id相同序列
                    pass
            else:
                seq_id_2_seq[seq_id] = [seq, dtype]
                # 重命名id
                seq_id_rename.append(["seq_%d" % seq_id_rename_idx, seq_id, seq])
                rename_seq_ids.add("seq_%d" % seq_id_rename_idx)
                if dtype == "positive":
                    rename_positives.append(["seq_%d" % seq_id_rename_idx, seq])
                seq_id_rename_idx += 1
            if seq in seq_2_seq_id:
                if seq_2_seq_id[seq][0] != seq_id:
                    print(seq_id, dtype, ":", seq)
                    print(seq_2_seq_id[seq][0], seq_2_seq_id[seq][1], ":", seq)
                    print("-" * 50)

            else:
                seq_2_seq_id[seq] = [seq_id, dtype]
            if "negative" in dirpath:
                negative_seq_ids_set.add(seq_id)
                negative_seqs_set.add(seq)
                negative_sample_num += 1
            else:
                positive_seq_ids_set.add(seq_id)
                positive_seqs_set.add(seq)
                positive_sample_num += 1

print("seq_ids_set: %d, seqs_set: %d, sample_num: %d" % (len(seq_ids_set), len(seqs_set), sample_num))
print("seq_id_2_seq: %d, seq_2_seq_id: %d" % (len(seq_id_2_seq), len(seq_2_seq_id)))
print("rename_seq_ids: %d, seq_id_rename: %d" % (len(rename_seq_ids), len(seq_id_rename)))
print("negative: seq_ids_set: %d, seqs_set: %d, sample_num: %d" % (len(negative_seq_ids_set), len(negative_seqs_set), negative_sample_num))
print("positive: seq_ids_set: %d, seqs_set: %d, sample_num: %d" % (len(positive_seq_ids_set), len(positive_seqs_set), positive_sample_num))
print("intersection: %d" % len(negative_seq_ids_set.intersection(positive_seq_ids_set)))
print("intersection: %d" % len(negative_seqs_set.intersection(positive_seqs_set)))

csv_writer(
    dataset=seq_id_rename,
    handle="../../../data/fasta/seq_id_rename.csv",
    header=["seq_id", "ori_seq_id", "seq"]
)

write_fasta(filepath="../../../data/fasta/seq_id_rename_positives.fasta", sequences=rename_positives)
rename_positive_seq_ids = set([item[0] for item in rename_positives])
rename_positive_seqs = set([item[1] for item in rename_positives])
print("rename_positives: %d, rename_positive_seq_ids: %d, rename_positive_seqs: %d" % (len(rename_positives), len(rename_positive_seq_ids), len(rename_positive_seqs)))
rename_positive_seq_lens = [len(seq) for seq in rename_positive_seqs]
print("positive seq len: 25%%: %d, 40%%: %d, 50%%: %d, 60%%: %d, 75%%: %d, 80%%: %d, 85%%: %d, 90%%: %d, 95%%: %d, 99%%: %d, median: %d, mean: %f, max: %d, min: %d" % (
    np.percentile(rename_positive_seq_lens, 25),
    np.percentile(rename_positive_seq_lens, 40),
    np.percentile(rename_positive_seq_lens, 50),
    np.percentile(rename_positive_seq_lens, 60),
    np.percentile(rename_positive_seq_lens, 75),
    np.percentile(rename_positive_seq_lens, 80),
    np.percentile(rename_positive_seq_lens, 85),
    np.percentile(rename_positive_seq_lens, 90),
    np.percentile(rename_positive_seq_lens, 95),
    np.percentile(rename_positive_seq_lens, 99),
    np.median(rename_positive_seq_lens),
    np.mean(rename_positive_seq_lens),
    np.max(rename_positive_seq_lens),
    np.min(rename_positive_seq_lens)
))

"""
seq_ids_set: 4036365, seqs_set: 4027023, sample_num: 4046567
seq_id_2_seq: 4036365, seq_2_seq_id: 4027023
rename_seq_ids: 4036440, seq_id_rename: 4036440
negative: seq_ids_set: 3822172, seqs_set: 3812830, sample_num: 3832374
positive: seq_ids_set: 214193, seqs_set: 214193, sample_num: 214193
intersection: 0
intersection: 0
rename_positives: 214193, rename_positive_seq_ids: 214193, rename_positive_seqs: 214193
positive seq len: 25%: 246, 40%: 265, 50%: 282, 60%: 318, 75%: 377, 80%: 449, 85%: 531, 90%: 627, 95%: 716, 99%: 798, median: 282, mean: 349.061244, max: 2986, min: 27
"""