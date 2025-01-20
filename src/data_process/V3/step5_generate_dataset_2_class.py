#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/13 16:08
@project: LucaPCycleV3
@file: step5_generate_dataset_2_class
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
去冗余都是50%
1. 负样本去冗余，正样本不去冗余，80%:10%:10%
2. 负样本去冗余，正样本去冗余，80%:10%:10%
3. 负样本去冗余，正样本去冗余，60%:20%:20%
4. 负样本去冗余，正样本不去冗余%:60%:20%:20%
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

    # 所有正样本
    all_positive_data_dir = "../../../data/fasta/positives/"
    assert os.path.exists(all_positive_data_dir)
    all_positive_num = 0
    # 所有正样本
    all_positives = []
    for filename in os.listdir(all_positive_data_dir):
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(all_positive_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                all_positives.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                all_positive_num += 1
    print("all_positive_num: %d, all_positives: %d" % (all_positive_num, len(all_positives)))
    for _ in range(10):
        random.shuffle(all_positives)

    # 50%去冗余正样本
    nonredundancy_positives = []
    nonredundancy_positive_data_dir = "/mnt/sanyuan.hy/workspace/cdhit/positive/50%/"
    assert os.path.exists(nonredundancy_positive_data_dir)
    nonredundancy_positive_num = 0
    nonredundancy_positive_filenames = sorted(os.listdir(nonredundancy_positive_data_dir))
    print("nonredundancy_positive_filenames:")
    print(nonredundancy_positive_filenames)
    for filename in nonredundancy_positive_filenames:
        if filename.endswith(".fasta"):
            for row in fasta_reader(os.path.join(nonredundancy_positive_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                nonredundancy_positives.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 1])
                nonredundancy_positive_num += 1
    print("nonredundancy_positive_num: %d, nonredundancy_positives: %d" % (nonredundancy_positive_num, len(nonredundancy_positives)))
    for _ in range(10):
        random.shuffle(nonredundancy_positives)

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
    return all_positives, nonredundancy_positives, nonredundancy_negatives


def generate_dataset(case_no, positives, negatives, rate, dataset_subfix):
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


def generate_dataset_test(positives, negatives):
    dataset_types = ["train", "dev", "test"]
    datasets = [[], [], []]
    print("-" * 50)
    print("case: test")
    # positives
    total_positives_num = len(positives)
    print("total_positives_num: %d" % total_positives_num)
    datasets[1].extend(positives[0: 10])
    datasets[2].extend(positives[10: 20])
    datasets[0].extend(positives[20: 40])
    print("train: %d, dev: %d, test: %d" % (len(datasets[0]), len(datasets[1]), len(datasets[2])))

    # negatives
    total_negatives_num = len(negatives)
    print("total_negatives_num: %d" % total_negatives_num)
    datasets[1].extend(negatives[0: 10])
    datasets[2].extend(negatives[10: 20])
    datasets[0].extend(negatives[20: 100])
    print("train: %d, dev: %d, test: %d" % (len(datasets[0]), len(datasets[1]), len(datasets[2])))
    for _ in range(10):
        random.shuffle(datasets[0])

    print("train: %d" % len(datasets[0]))
    print("dev: %d" % len(datasets[1]))
    print("test: %d" % len(datasets[2]))

    dataset_dirpath = "../../../dataset/extra_p_2_class_v3_test/protein/binary_class"
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
    all_positives, nonredundancy_positives, nonredundancy_negatives = load_data()
    """
    去冗余都是50%
    1. 负样本去冗余，正样本不去冗余，80%:10%:10%
    2. 负样本去冗余，正样本去冗余，80%:10%:10%
    3. 负样本去冗余，正样本去冗余，60%:20%:20%
    4. 负样本去冗余，正样本不去冗余%:60%:20%:20%
    """
    generate_dataset_test(
        positives=nonredundancy_positives,
        negatives=nonredundancy_negatives
    )

    generate_dataset(
        case_no=1,
        positives=all_positives,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_01"
    )
    generate_dataset(
        case_no=2,
        positives=nonredundancy_positives,
        negatives=nonredundancy_negatives,
        rate=0.8,
        dataset_subfix="case_02"
    )
    generate_dataset(
        case_no=3,
        positives=nonredundancy_positives,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_03"
    )
    generate_dataset(
        case_no=13,
        positives=all_positives,
        negatives=nonredundancy_negatives,
        rate=0.6,
        dataset_subfix="case_04"
    )
