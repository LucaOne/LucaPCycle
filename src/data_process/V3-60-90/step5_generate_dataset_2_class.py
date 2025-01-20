#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/13 16:08
@project: LucaPCycleV3
@file: generate_dataset_2_class
@desc: generate dataset for 2 class
'''
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
"""
    负样本去冗余都是50%，正样本去冗余60%, 70%, 80%, 90%
    5. 负样本去冗余50%，正样本去冗余60%，80%:10%:10%
    6. 负样本去冗余50%，正样本去冗余60%，60%:20%:20%
    7. 负样本去冗余50%，正样本去冗余70%，80%:10%:10%
    8. 负样本去冗余50%，正样本去冗余70%，60%:20%:20%
    9. 负样本去冗余50%，正样本去冗余80%，80%:10%:10%
    10. 负样本去冗余50%，正样本去冗余80%，60%:20%:20%
    11. 负样本去冗余50%，正样本去冗余90%，80%:10%:10%
    12. 负样本去冗余50%，正样本去冗余90%，60%:20%:20%
"""


def load_data():
    # seq_id 重命名
    data_filepath = "../../../data/fasta/seq_id_rename.csv"
    assert os.path.exists(data_filepath)
    seq_id_rename = {}
    for row in csv_reader(data_filepath, header=True, header_filter=True):
        seq_id, ori_seq_id, seq = row[0], row[1], row[2]
        if ori_seq_id not in seq_id_rename:
            seq_id_rename[ori_seq_id] = {}
            seq_id_rename[ori_seq_id][seq] = seq_id
        elif seq not in seq_id_rename[ori_seq_id]:
            seq_id_rename[ori_seq_id][seq] = seq_id
        else:
            raise Exception(row)

    # 60%去冗余正样本
    nonredundancy_positives_60 = []
    nonredundancy_positive_60_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/positive/60%/"
    assert os.path.exists(nonredundancy_positive_60_data_dir)
    nonredundancy_positive_60_num = 0
    nonredundancy_positive_60_filenames = sorted(os.listdir(nonredundancy_positive_60_data_dir))
    print("nonredundancy_positive_60_filenames:")
    print(nonredundancy_positive_60_filenames)
    for filename in nonredundancy_positive_60_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_positive_60_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                nonredundancy_positives_60.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                nonredundancy_positive_60_num += 1
    print("nonredundancy_positive_60_num: %d, nonredundancy_positives_60: %d" % (nonredundancy_positive_60_num, len(nonredundancy_positives_60)))
    for _ in range(10):
        random.shuffle(nonredundancy_positives_60)

    # 70%去冗余正样本
    nonredundancy_positives_70 = []
    nonredundancy_positive_70_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/positive/70%/"
    assert os.path.exists(nonredundancy_positive_70_data_dir)
    nonredundancy_positive_70_num = 0
    nonredundancy_positive_70_filenames = sorted(os.listdir(nonredundancy_positive_70_data_dir))
    print("nonredundancy_positive_70_filenames:")
    print(nonredundancy_positive_70_filenames)
    for filename in nonredundancy_positive_70_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_positive_70_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                nonredundancy_positives_70.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                nonredundancy_positive_70_num += 1
    print("nonredundancy_positive_70_num: %d, nonredundancy_positives_70: %d" % (nonredundancy_positive_70_num, len(nonredundancy_positives_70)))
    for _ in range(10):
        random.shuffle(nonredundancy_positives_70)

    # 80%去冗余正样本
    nonredundancy_positives_80 = []
    nonredundancy_positive_80_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/positive/80%/"
    assert os.path.exists(nonredundancy_positive_80_data_dir)
    nonredundancy_positive_80_num = 0
    nonredundancy_positive_80_filenames = sorted(os.listdir(nonredundancy_positive_80_data_dir))
    print("nonredundancy_positive_80_filenames:")
    print(nonredundancy_positive_80_filenames)
    for filename in nonredundancy_positive_80_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_positive_80_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                nonredundancy_positives_80.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                nonredundancy_positive_80_num += 1
    print("nonredundancy_positive_80_num: %d, nonredundancy_positives_80: %d" % (nonredundancy_positive_80_num, len(nonredundancy_positives_80)))
    for _ in range(10):
        random.shuffle(nonredundancy_positives_80)

    # 90%去冗余正样本
    nonredundancy_positives_90 = []
    nonredundancy_positive_90_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/positive/90%/"
    assert os.path.exists(nonredundancy_positive_90_data_dir)
    nonredundancy_positive_90_num = 0
    nonredundancy_positive_90_filenames = sorted(os.listdir(nonredundancy_positive_90_data_dir))
    print("nonredundancy_positive_90_filenames:")
    print(nonredundancy_positive_90_filenames)
    for filename in nonredundancy_positive_90_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_positive_90_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                nonredundancy_positives_90.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                nonredundancy_positive_90_num += 1
    print("nonredundancy_positive_90_num: %d, nonredundancy_positives_90: %d" % (nonredundancy_positive_90_num, len(nonredundancy_positives_90)))
    for _ in range(10):
        random.shuffle(nonredundancy_positives_90)

    # 50%去冗余负样本
    nonredundancy_negatives = []
    nonredundancy_negative_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/negative/5_cdhit50%"
    assert os.path.exists(nonredundancy_negative_data_dir)
    nonredundancy_negative_num = 0
    exists_nonredundancy_negative_seqs = set()
    nonredundancy_negative_filenames = sorted(os.listdir(nonredundancy_negative_data_dir))
    print("nonredundancy_negative_filenames:")
    print(nonredundancy_negative_filenames)
    for filename in nonredundancy_negative_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_negative_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                # 重复序列不考虑
                if seq not in exists_nonredundancy_negative_seqs:
                    nonredundancy_negatives.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 0])
                    exists_nonredundancy_negative_seqs.add(seq)
                nonredundancy_negative_num += 1
    for _ in range(10):
        random.shuffle(nonredundancy_negatives)
    print("nonredundancy_negative_num: %d, nonredundancy_negatives: %d, exists_nonredundancy_negative_seqs: %d" % (nonredundancy_negative_num, len(nonredundancy_negatives), len(exists_nonredundancy_negative_seqs)))

    return nonredundancy_positives_60, nonredundancy_positives_70, nonredundancy_positives_80, nonredundancy_positives_90, nonredundancy_negatives


def generate_dataset(
        case_no,
        positives,
        negatives,
        rate,
        dataset_subfix
):
    dataset_types = ["train", "dev", "test"]
    datasets = [[], [], []]
    print("case: %d" % case_no)
    # positives
    total_positives_num = len(positives)
    print("total_positives_num: %d" % total_positives_num)
    dev_test_positives_num = int((1 - rate)/2 * total_positives_num)
    print("dev_test_positives_num: %d" % dev_test_positives_num)
    datasets[1].extend(positives[0: dev_test_positives_num])
    datasets[2].extend(positives[dev_test_positives_num: dev_test_positives_num + dev_test_positives_num])
    datasets[0].extend(positives[dev_test_positives_num + dev_test_positives_num:])
    print("train: %d, dev: %d, test: %d" % (len(datasets[0]), len(datasets[1]), len(datasets[2])))
    assert total_positives_num == len(datasets[0]) + len(datasets[1]) + len(datasets[2])

    # negatives
    total_negatives_num = len(negatives)
    print("total_negatives_num: %d" % total_negatives_num)
    dev_test_negatives_num = int((1 - rate)/2 * total_negatives_num)
    print("dev_test_negatives_num: %d" % dev_test_negatives_num)
    datasets[1].extend(negatives[0: dev_test_negatives_num])
    datasets[2].extend(negatives[dev_test_negatives_num: dev_test_negatives_num + dev_test_negatives_num])
    datasets[0].extend(negatives[dev_test_negatives_num + dev_test_negatives_num:])
    print("train: %d, dev: %d, test: %d" % (len(datasets[0]), len(datasets[1]), len(datasets[2])))
    assert total_positives_num + total_negatives_num == len(datasets[0]) + len(datasets[1]) + len(datasets[2])
    for _ in range(10):
        random.shuffle(datasets[0])

    print("train: %d" % len(datasets[0]))
    print("dev: %d" % len(datasets[1]))
    print("test: %d" % len(datasets[2]))

    seq_len_list = [len(item[2]) for item in datasets[0]]
    seq_len_list.extend([len(item[2]) for item in datasets[1]])
    seq_len_list.extend([len(item[2]) for item in datasets[2]])
    print("seq len: 25%%: %d, 40%%: %d, 50%%: %d, 60%%: %d, 75%%: %d, 80%%: %d, 85%%: %d, 90%%: %d, 95%%: %d, 99%%: %d, median: %d, mean: %f, max: %d, min: %d" % (
        np.percentile(seq_len_list, 25),
        np.percentile(seq_len_list, 40),
        np.percentile(seq_len_list, 50),
        np.percentile(seq_len_list, 60),
        np.percentile(seq_len_list, 75),
        np.percentile(seq_len_list, 80),
        np.percentile(seq_len_list, 85),
        np.percentile(seq_len_list, 90),
        np.percentile(seq_len_list, 95),
        np.percentile(seq_len_list, 99),
        np.median(seq_len_list),
        np.mean(seq_len_list),
        np.max(seq_len_list),
        np.min(seq_len_list)
    ))

    dataset_dirpath = "../../../dataset/extra_p_2_class_v3_%s/protein/binary_class" % dataset_subfix
    for dataset_type in dataset_types:
        dirpath = os.path.join(dataset_dirpath, dataset_type)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    assert not os.path.exists(os.path.join(dataset_dirpath, "label.txt"))
    csv_writer(
        dataset=[[0], [1]],
        handle=os.path.join(dataset_dirpath, "label.txt"),
        header=["label"]
    )

    assert not os.path.exists(os.path.join(dataset_dirpath, "train", "train.csv"))
    csv_writer(
        dataset=datasets[0],
        handle=os.path.join(dataset_dirpath, "train", "train.csv"),
        header=["seq_id", "seq_type", "seq", "label"])

    assert not os.path.exists(os.path.join(dataset_dirpath, "dev", "dev.csv"))
    csv_writer(
        dataset=datasets[1],
        handle=os.path.join(dataset_dirpath, "dev", "dev.csv"),
        header=["seq_id", "seq_type", "seq", "label"])

    assert not os.path.exists(os.path.join(dataset_dirpath, "test", "test.csv"))
    csv_writer(
        dataset=datasets[2],
        handle=os.path.join(dataset_dirpath, "test", "test.csv"),
        header=["seq_id", "seq_type", "seq", "label"])

    print("#" * 50)


if __name__ == "__main__":
    nonredundancy_positives_60, nonredundancy_positives_70, \
    nonredundancy_positives_80, nonredundancy_positives_90, nonredundancy_negatives = load_data()
    all_sequences = {}
    for item in nonredundancy_positives_60:
        all_sequences[item[0]] = item[2]
    print("add nonredundancy_positives_60: %d, all_sequences: %d" % (len(nonredundancy_positives_60), len(all_sequences)))
    for item in nonredundancy_positives_70:
        all_sequences[item[0]] = item[2]
    print("add nonredundancy_positives_70: %d, all_sequences: %d" % (len(nonredundancy_positives_70), len(all_sequences)))
    for item in nonredundancy_positives_80:
        all_sequences[item[0]] = item[2]
    print("add nonredundancy_positives_80: %d, all_sequences: %d" % (len(nonredundancy_positives_80), len(all_sequences)))
    for item in nonredundancy_positives_90:
        all_sequences[item[0]] = item[2]
    print("add nonredundancy_positives_90: %d, all_sequences: %d" % (len(nonredundancy_positives_90), len(all_sequences)))
    for item in nonredundancy_negatives:
        all_sequences[item[0]] = item[2]
    print("add nonredundancy_negatives: %d, all_sequences: %d" % (len(nonredundancy_negatives), len(all_sequences)))

    split_num = 4
    per_num = (len(all_sequences) + split_num - 1) // split_num
    all_sequences_filepath = "../../../dataset/fasta/cdhit-p60-90-n50/cdhit-p60-90-n50_seq.fasta"
    if not os.path.exists(os.path.dirname(all_sequences_filepath)):
        os.makedirs(os.path.dirname(all_sequences_filepath))
    save_dir = os.path.dirname(all_sequences_filepath)
    filename = os.path.basename(all_sequences_filepath)
    print("total: %d, per_num: %d" % (len(all_sequences), per_num))
    all_sequences = [[item[0], item[1]] for item in all_sequences.items()]
    for idx in range(split_num):
        strs = filename.split(".")
        cur_filename = ".".join(strs[0:-1]) + "_part_%02d_%02d." % (idx + 1, split_num) + strs[-1]
        save_path = os.path.join(save_dir, cur_filename)
        cur_dataset = all_sequences[idx * per_num: min(len(all_sequences), (idx + 1) * per_num)]
        print("dataset: %d" % len(cur_dataset))
        write_fasta(save_path, cur_dataset)
        print("done %d/%d" % (idx + 1, split_num))

    """
    负样本去冗余都是50%，正样本去冗余60%, 70%, 80%, 90%
    5. 负样本去冗余50%，正样本去冗余60%，80%:10%:10%
    6. 负样本去冗余50%，正样本去冗余60%，60%:20%:20%
    7. 负样本去冗余50%，正样本去冗余70%，80%:10%:10%
    8. 负样本去冗余50%，正样本去冗余70%，60%:20%:20%
    9. 负样本去冗余50%，正样本去冗余80%，80%:10%:10%
    10. 负样本去冗余50%，正样本去冗余80%，60%:20%:20%
    11. 负样本去冗余50%，正样本去冗余90%，80%:10%:10%
    12. 负样本去冗余50%，正样本去冗余90%，60%:20%:20%
    """
    generate_dataset(
        case_no=5,
        positives=nonredundancy_positives_60,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_05"
    )
    generate_dataset(
        case_no=6,
        positives=nonredundancy_positives_60,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_06"
    )
    generate_dataset(
        case_no=7,
        positives=nonredundancy_positives_70,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_07"
    )
    generate_dataset(
        case_no=8,
        positives=nonredundancy_positives_70,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_08"
    )
    generate_dataset(
        case_no=9,
        positives=nonredundancy_positives_80,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_09"
    )
    generate_dataset(
        case_no=10,
        positives=nonredundancy_positives_80,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_10"
    )

    generate_dataset(
        case_no=11,
        positives=nonredundancy_positives_90,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_11"
    )
    generate_dataset(
        case_no=12,
        positives=nonredundancy_positives_90,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_12"
    )


