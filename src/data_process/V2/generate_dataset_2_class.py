#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/13 16:08
@project: LucaPCycle
@file: generate_dataset_2_class
@desc: generate dataset for 2 class
'''

import os
import random

from src.file_operator import csv_reader, csv_writer, fasta_reader

# 所有正样本+50中取负样本
positives = []

prot_id = 0
positive_datadir = "../../../data/31P_genes/"
for filename in os.listdir(positive_datadir):
    if filename.endswith(".faa"):
        for row in fasta_reader(os.path.join(positive_datadir, filename)):
            prot_id += 1
            positives.append([filename.replace(".faa", "") + "_" + str(prot_id), "prot", row[1].strip().upper(), 1])
for _ in range(10):
    random.shuffle(positives)

negatives = []
prot_id = 0
for row in csv_reader("../../../data/cold_spring_sample_50.csv"):
    seq_id, seq_type, seq, label = row
    if label != "Positive":
        prot_id += 1
        negatives.append(["neg_" + str(prot_id), "prot", seq.strip().upper(), 0])
for _ in range(10):
    random.shuffle(negatives)
print("positives: %d, negative: %d" % (len(positives), len(negatives)))

dataset_types = ["train", "dev", "test"]
datasets = [[], [], []]

rate = 0.99
pos_size = int(len(positives) * rate)
datasets[0].extend(positives[:pos_size])
dev_size = (len(positives) - pos_size) // 2
datasets[1].extend(positives[pos_size:pos_size+dev_size])
datasets[2].extend(positives[pos_size+dev_size:])


neg_size = int(len(negatives) * rate)
datasets[0].extend(negatives[:neg_size])
dev_size = (len(negatives) - neg_size) // 2
datasets[1].extend(negatives[neg_size:neg_size+dev_size])
datasets[2].extend(negatives[neg_size+dev_size:])

for _ in range(10):
    random.shuffle(datasets[0])
print("train: %d" % len(datasets[0]))
print("dev: %d" % len(datasets[1]))
print("test: %d" % len(datasets[2]))

dataset_dirpath = "../../../dataset/extra_p_2_class/protein/binary_class"
for dataset_type in dataset_types:
    dirpath = os.path.join(dataset_dirpath, dataset_type)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

csv_writer(
    dataset=[[0], [1]],
    handle=os.path.join(dataset_dirpath, "label.txt"),
    header=["label"]
)

csv_writer(
    dataset=datasets[0],
    handle=os.path.join(dataset_dirpath, "train", "train.csv"),
    header=["seq_id", "seq_type", "seq", "label"])

csv_writer(
    dataset=datasets[1],
    handle=os.path.join(dataset_dirpath, "dev", "dev.csv"),
    header=["seq_id", "seq_type", "seq", "label"])

csv_writer(
    dataset=datasets[2],
    handle=os.path.join(dataset_dirpath, "test", "test.csv"),
    header=["seq_id", "seq_type", "seq", "label"])

'''
positives: 214193, negative: 751331
train: 955868
dev: 4828
test: 4828


(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' Uniprot_neg.faa | wc -l
1000000
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' 31P_gene.faa | wc -l
214193
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' intra_p.fasta | wc -l
664,615
664573
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' N.fasta  | wc -l
273,824
273757
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' other.fasta   | wc -l
1,270,613 
1270612
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' S.fasta   | wc -l
920,822
920115

-- where length(sequence) > 0
S	920115
intra_P	664573
N	273757
other	1270612
Positive	214193

other.fasta : 1270612
--------------------
cold_spring_stats.ipynb : 1
--------------------
S.fasta : 920115
--------------------
N.fasta : 273757
--------------------
intra_p.fasta : 664573
--------------------
Uniprot_neg.faa : 1000000
--------------------
31P_gene.faa : 214193
--------------------

-- S	920322
-- intra_P	664615
-- N	273824
-- other	1270613
-- Positive	214193

(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' 31P_gene50.faa    | wc -l
14322
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' intra_p50.faa    | wc -l
36661
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' N50.faa    | wc -l
8447
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' other50.faa    | wc -l
63606
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' S50.faa    | wc -l
63525
(lucaone_tasks) sanyuan.hy@hey:/mnt/sanyuan.hy/cindy/冷泉$ grep  '>' Uniprot_neg.faa50.faa    | wc -l
579092
'''