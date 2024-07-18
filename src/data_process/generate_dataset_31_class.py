#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/15 16:54
@project: LucaPCycle
@file: generate_dataset_31_class
@desc: generate dataset for 31 class
'''
import sys, os, random
sys.path.append("")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from file_operator import csv_reader, csv_writer, fasta_reader
    from utils import load_labels, save_labels
except ImportError as e:
    from src.file_operator import csv_reader, csv_writer, fasta_reader
    from src.utils import load_labels, save_labels


def generate_dataset_multi_class(dirpath, save_path, train_rate=0.9):
    # 每一个文件便是一个类别
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
            seq_id = row[0].split(" ")[0].strip()
            if seq_id[0] == ">":
                seq_id = seq_id[1:]
            seq = row[1].strip().upper()
            dataset[label_name].append([seq_id, "prot", seq, label_list.index(label_name)])

    train_set = []
    dev_set = []
    test_set = []
    train_sample_stats = [0] * len(dataset)
    for item in dataset.items():
        label_name = item[0]
        label_idx = label_list.index(label_name)
        cur = item[1]
        for _ in range(10):
            random.shuffle(cur)
        cur_size = len(cur)
        cur_train_size = int(cur_size * train_rate)
        train_sample_stats[label_idx] = cur_train_size
        train_set.extend(cur[0:cur_train_size])
        cur_dev_size = int((0.5 - train_rate/2) * cur_size)
        dev_set.extend(cur[cur_train_size:cur_train_size + cur_dev_size])
        test_set.extend(cur[cur_train_size + cur_dev_size:])
        print("label_name: %s, train: %d, dev: %d, test: %d" % (label_name,
                                                                cur_train_size,
                                                                cur_dev_size,
                                                                cur_size - cur_train_size - cur_dev_size))
    dataset_type_list = ["train", "dev", "test"]
    for dataset_type in dataset_type_list:
        path = os.path.join(save_path, dataset_type)
        if not os.path.exists(path):
            os.makedirs(path)
    for _ in range(10):
        random.shuffle(train_set)
    csv_writer([[v] for v in label_list], os.path.join(save_path, "label.txt"), header=["label"])
    print("train_set size: %d" % len(train_set))
    csv_writer(
        dataset=train_set,
        handle=os.path.join(save_path, "train", "train.csv"),
        header=["seq_id", "seq_type", "seq", "label"]
    )

    print("dev_set size: %d" % len(dev_set))
    csv_writer(
        dataset=dev_set,
        handle=os.path.join(save_path, "dev", "dev.csv"),
        header=["seq_id", "seq_type", "seq", "label"]
    )

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
    save_path = "../../dataset/extra_p_31_class_v2/protein/multi_class/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generate_dataset_multi_class(
        dirpath="../../data/31P_genes/",
        save_path=save_path,
        train_rate=0.9
    )
'''
# stats
# stats
# seq len: [24, 33423, 605.4188537761114]
# [1937, 2835, 1243, 21613, 64, 1854, 38405, 2952, 3973, 2222, 6768, 31833, 4680, 1217, 9263, 9049, 3340, 603, 5413, 12970, 10106, 213, 1279, 3139, 11330, 347, 1418, 1369, 104, 1040, 179]
# [19.827052142488384, 13.54673721340388, 30.89702333065165, 1.776939804747143, 600.078125, 20.71467098166127, 1.0, 13.009823848238483, 9.666498867354644, 17.283978397839785, 5.674497635933806, 1.2064524235855874, 8.206196581196581, 31.55710764174199, 4.1460649897441435, 4.244115371864295, 11.498502994011975, 63.68988391376451, 7.094956585996675, 2.9610639938319196, 3.800217692459925, 180.30516431924883, 30.027365129007038, 12.234788149092067, 3.389673433362754, 110.67723342939482, 27.08392101551481, 28.053323593864135, 369.27884615384613, 36.92788461538461, 214.55307262569832]'''