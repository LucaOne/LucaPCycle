#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/9/18 17:31
@project: LucaPCycleV3
@file: step3_split_fasta
@desc: xxxx
"""
import os
import sys
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import *
    from utils import clean_special_char
except ImportError:
    from src.file_operator import *
    from src.utils import clean_special_char

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, default=None, required=True, help="filepath")
parser.add_argument("--split_num", default=None, type=int, required=True, help="split num")
args = parser.parse_args()

assert os.path.exists(args.filepath)
filename = os.path.basename(args.filepath)
for idx in range(args.split_num):
    strs = filename.split(".")
    cur_filename = ".".join(strs[0:-1]) + "_part_%02d_%02d." % (idx + 1, args.split_num) + strs[-1]
    assert not os.path.exists(cur_filename)

assert args.split_num >= 1
dataset = []
for row in fasta_reader(args.filepath):
    seq_id = row[0].strip()
    seq = clean_special_char(row[0], row[1])
    dataset.append([seq_id, seq])
per_num = (len(dataset) + args.split_num - 1) // args.split_num
save_dir = os.path.dirname(args.filepath)
filename = os.path.basename(args.filepath)
print("total: %d, per_num: %d" % (len(dataset), per_num))
for idx in range(args.split_num):
    strs = filename.split(".")
    cur_filename = ".".join(strs[0:-1]) + "_part_%02d_%02d." % (idx + 1, args.split_num) + strs[-1]
    save_path = os.path.join(save_dir, cur_filename)
    cur_dataset = dataset[idx * per_num: min(len(dataset), (idx + 1) * per_num)]
    print("dataset: %d" % len(cur_dataset))
    write_fasta(save_path, cur_dataset)
    print("done %d/%d" % (idx + 1, args.split_num))

"""
python step3_split_fasta.py --filepath ../../../data/fasta/seq_id_rename_positives.fasta --split_num 4
total: 214193
dataset: 53549
done 1/4
dataset: 53549
done 2/4
dataset: 53549
done 3/4
dataset: 53546
done 4/4

python step3_split_fasta.py --filepath ../../../data/fasta/seq_id_rename_negatives.fasta --split_num 4
total: 3822247
dataset: 955562
done 1/4
dataset: 955562
done 2/4
dataset: 955562
done 3/4
dataset: 955561
done 4/4
"""