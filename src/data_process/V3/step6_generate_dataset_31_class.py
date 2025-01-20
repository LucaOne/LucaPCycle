#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/15 16:54
@project: LucaPCycleV3
@file: step6_generate_dataset_31_class
@desc: generate dataset for 31 class
'''
import numpy as np
import sys, os, random
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from file_operator import csv_reader, csv_writer, fasta_reader
    from utils import load_labels, save_labels, clean_special_char
except ImportError as e:
    from src.file_operator import csv_reader, csv_writer, fasta_reader
    from src.utils import load_labels, save_labels, clean_special_char


def generate_dataset_multi_class(dirpath, save_path, train_rate=0.8):
    # seq_id 重命名
    data_filepath = "../../../data/fasta/seq_id_rename.csv"
    assert os.path.join(data_filepath)
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

    # 每一个文件便是一个类别
    assert os.path.exists(dirpath)
    dataset = {}
    label_list = []
    for filename in os.listdir(dirpath):
        if not filename.endswith(".faa"):
            continue
        label_name = filename.replace(".faa", "")
        label_list.append(label_name)
        if label_name not in dataset:
            dataset[label_name] = []
        filepath = os.path.join(dirpath, filename)
        for row in fasta_reader(filepath):
            ori_seq_id = row[0].split(" ")[0].strip()
            if ori_seq_id[0] == ">":
                ori_seq_id = ori_seq_id[1:]
            seq = clean_special_char(ori_seq_id, row[1])
            dataset[label_name].append([seq_id_rename[ori_seq_id][seq], "prot", seq, label_list.index(label_name)])

    train_set = []
    dev_set = []
    test_set = []
    train_sample_stats = [0] * len(dataset)
    for item in dataset.items():
        label_name = item[0]
        label_idx = label_list.index(label_name)
        cur = item[1]
        for _ in range(20):
            random.shuffle(cur)
        cur_size = len(cur)
        cur_dev_test_size = int(cur_size * (1 - train_rate)/2)
        train_sample_stats[label_idx] = cur_size - 2 * cur_dev_test_size
        train_set.extend(cur[cur_dev_test_size + cur_dev_test_size:])
        dev_set.extend(cur[0: cur_dev_test_size])
        test_set.extend(cur[cur_dev_test_size: cur_dev_test_size + cur_dev_test_size])
        print("label_name: %s, train: %d, dev: %d, test: %d" % (label_name,
                                                                cur_size - cur_dev_test_size - cur_dev_test_size,
                                                                cur_dev_test_size,
                                                                cur_dev_test_size))
    dataset_type_list = ["train", "dev", "test"]
    for dataset_type in dataset_type_list:
        path = os.path.join(save_path, dataset_type)
        if not os.path.exists(path):
            os.makedirs(path)
    for _ in range(10):
        random.shuffle(train_set)

    seq_len_list = [len(item[2]) for item in train_set]
    seq_len_list.extend([len(item[2]) for item in dev_set])
    seq_len_list.extend([len(item[2]) for item in test_set])
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

    assert not os.path.exists(os.path.join(save_path, "label.txt"))
    csv_writer([[v] for v in label_list], os.path.join(save_path, "label.txt"), header=["label"])

    assert not os.path.exists(os.path.join(save_path, "train", "train.csv"))
    print("train_set size: %d" % len(train_set))
    csv_writer(
        dataset=train_set,
        handle=os.path.join(save_path, "train", "train.csv"),
        header=["seq_id", "seq_type", "seq", "label"]
    )

    assert not os.path.exists(os.path.join(save_path, "dev", "dev.csv"))
    print("dev_set size: %d" % len(dev_set))
    csv_writer(
        dataset=dev_set,
        handle=os.path.join(save_path, "dev", "dev.csv"),
        header=["seq_id", "seq_type", "seq", "label"]
    )

    assert not os.path.exists(os.path.join(save_path, "test", "test.csv"))
    print("test_set size: %d" % len(test_set))
    csv_writer(
        dataset=test_set,
        handle=os.path.join(save_path, "test", "test.csv"),
        header=["seq_id", "seq_type", "seq", "label"]
    )
    max_label_cnt = max(train_sample_stats)
    print("train_sample_stats:")
    print(train_sample_stats)
    print([max_label_cnt/v for v in train_sample_stats])


if __name__ == "__main__":
    save_path = "../../../dataset/extra_p_31_class_v3/protein/multi_class/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generate_dataset_multi_class(
        dirpath="../../../data/31P_genes/fasta/",
        save_path=save_path,
        train_rate=0.8
    )
    generate_dataset_multi_class(
        dirpath="../../../data/31P_genes/fasta/",
        save_path=save_path,
        train_rate=0.6
    )

