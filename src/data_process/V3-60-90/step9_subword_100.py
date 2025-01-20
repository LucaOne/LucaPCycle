#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/15 13:56
@project: LucaPCycleV3
@file: subword for all positives
@desc: apply BPE algorithm for subword operation
'''
import os
import sys
import codecs
import random
import argparse
from subword_nmt.get_vocab import get_vocab
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader, txt_reader, txt_writer
    from utils import clean_special_char
except ImportError as e:
    from src.file_operator import fasta_reader, csv_reader, txt_reader, txt_writer
    from src.utils import clean_special_char


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", default="learn_bpe", type=str, required=True, choices=["corpus",
                                                                                         "learn_bpe",
                                                                                         "apply_bpe",
                                                                                         "learn_joint_bpe_and_vocab",
                                                                                         "tokenize",
                                                                                         "get_vocab",
                                                                                         "subword_vocab_2_token_vocab"
                                                                                         ],
                        help="subword running type.")
    '''For corpus'''
    '''For Learn and Apply'''
    parser.add_argument("--infile",  type=str, default=None, help="corpus")
    parser.add_argument("--outfile",  type=str, default=None, help="output filepath")

    '''For Learn'''
    parser.add_argument("--min_frequency",  type=int, default=2, help="min frequency")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--is_dict", action="store_true", help="is dict")
    parser.add_argument("--num_symbols", type=int, default=20000, help="the specified vocab size")
    parser.add_argument("--num_workers", type=int, default=8, help="worker number")

    '''For Tokenize and Apply'''
    parser.add_argument("--codes_file",  type=str, default=None, help="subword codes filepath")
    '''For Tokenize '''
    parser.add_argument("--seq",  type=str, default=None, help="the sequence that want to tokenize")

    args = parser.parse_args()
    return args


def generate_corpus(save_filepath, rate=0.2):
    '''
    fasta to sequence corpus
    :param save_filepath
    :param rate
    :return:
    '''

    # 所有去冗余后的正样本 + 每类去冗余后的负样本20%
    corpus = []

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
    corpus.extend(all_positives)
    print("label: all_positives, size: %d, selected size: %d" % (len(all_positives), len(all_positives)))

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
            cur_nonredundancy_negative_num = 0
            cur_nonredundancy_negatives = []
            for row in fasta_reader(os.path.join(nonredundancy_negative_data_dir, filename)):
                ori_seq_id, seq = row[0].strip(), clean_special_char(row[0], row[1])
                if ori_seq_id[0] == ">":
                    ori_seq_id = ori_seq_id[1:]
                # 重复序列不考虑
                if seq not in exists_nonredundancy_negative_seqs:
                    nonredundancy_negatives.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 0])
                    cur_nonredundancy_negatives.append([seq_id_rename[ori_seq_id][seq], "prot", seq, 0])
                    exists_nonredundancy_negative_seqs.add(seq)
                nonredundancy_negative_num += 1
                cur_nonredundancy_negative_num += 1
            for _ in range(10):
                random.shuffle(cur_nonredundancy_negatives)
            selected_size = int(len(cur_nonredundancy_negatives) * rate)
            corpus.extend(cur_nonredundancy_negatives[0: selected_size])
            print("label: %s, size: %d, selected size: %d" % (filename, cur_nonredundancy_negative_num, selected_size))
    print("nonredundancy_negative_num: %d, nonredundancy_negatives: %d" % (nonredundancy_negative_num, len(nonredundancy_negatives)))

    print("corpus: %d" % len(corpus))
    with open(save_filepath, "w") as wfp:
        for item in corpus:
            wfp.write(item[2] + "\n")


def learn(args):
    learn_bpe(infile=open(args.infile, "r"), outfile=open(args.outfile, "w"),
              min_frequency=args.min_frequency, verbose=args.verbose,
              is_dict=args.is_dict, num_symbols=args.num_symbols, num_workers=args.num_workers)


def apply(args):
    bpe_codes = codecs.open(args.codes_file)
    bpe = BPE(codes=bpe_codes)
    bpe.process_lines(args.infile, open(args.outfile, "w"), num_workers=args.num_workers)


def vocab(args):
    get_vocab(open(args.infile, "r"), open(args.outfile, "w"))


def subword_vocab_2_token_vocab(args):
    '''
    transform subword results into vocab
    :param args:
    :return:
    '''
    vocabs = set()
    with open(args.infile, "r") as rfp:
        for line in rfp:
            v = line.strip().split()[0].replace("@@", "")
            vocabs.add(v)
    vocabs = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + sorted(list(vocabs), key=lambda x: (len(x), x))
    with open(args.outfile, "w") as wfp:
        for v in vocabs:
            wfp.write(v + "\n")


def tokenize(args):
    bpe_codes = codecs.open(args.codes_file)
    bpe = BPE(bpe_codes, merges=-1, separator='')
    p = bpe.process_line(args.seq).split()
    return p


if __name__ == "__main__":
    args = get_args()
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr, value))
    if args.func != "tokenize":
        dir_path = os.path.dirname(args.outfile)
        if not os.path.exists(dir_path):
            print("Warning: output dir %s not exists, created!" % dir_path)
            os.makedirs(dir_path)
    if args.func == "corpus":
        generate_corpus(save_filepath=args.outfile, rate=0.2)
    elif args.func == "learn_bpe":
        if ".fa" in args.infile:
            # transform fasta to corpus
            print('fasta convect to corpus txt')
            save_path = os.path.join(os.path.dirname(args.infile), ".".join(os.path.basename(args.infile).split(".")[0: -1]) + ".txt")
            if os.path.exists(save_path):
                raise Exception("Save path :%s exsits!" % save_path)
            generate_corpus(save_filepath=save_path, rate=0.2)
            args.infile = save_path
        learn(args)
    elif args.func == "tokenize":
        print("input seq:")
        print(args.seq)
        print("input seq size:")
        print(len(args.seq))
        token = tokenize(args)
        print("seq tokenize output:")
        print(token)
        print("seq tokenize size:")
        print(len(token))
    elif args.func == "apply_bpe":
        if ".fas" in args.infile:
            # fasta，not sequence tokenization corpus
            raise Exception("the input file is fasta，not sequence tokenization corpus")
        apply(args)
    elif args.func == "get_vocab":
        vocab(args)
    elif args.func == "subword_vocab_2_token_vocab":
        subword_vocab_2_token_vocab(args)

    '''
    这里使用的是所有(100)正样本
    all_positive_num: 214193, all_positives: 214193
    label: all_positives, size: 214193, selected size: 214193
    nonredundancy_negative_filenames:
    ['lucapcycle_cold_spring_negatives_N_filtered90%_cdhit50.fasta', 'lucapcycle_cold_spring_negatives_S_filtered90%_cdhit50.fasta', 'lucapcycle_cold_spring_negatives_intra_P_filtered90%_cdhit50.fasta', 'lucapcycle_cold_spring_negatives_other_filtered90%_cdhit50.fasta', 'lucapcycle_cold_spring_negatives_uniprot_filtered90%_cdhit50.fasta']
    label: lucapcycle_cold_spring_negatives_N_filtered90%_cdhit50.fasta, size: 9277, selected size: 1855
    label: lucapcycle_cold_spring_negatives_S_filtered90%_cdhit50.fasta, size: 64877, selected size: 12927
    label: lucapcycle_cold_spring_negatives_intra_P_filtered90%_cdhit50.fasta, size: 39048, selected size: 7463
    label: lucapcycle_cold_spring_negatives_other_filtered90%_cdhit50.fasta, size: 68524, selected size: 13699
    label: lucapcycle_cold_spring_negatives_uniprot_filtered90%_cdhit50.fasta, size: 673884, selected size: 134776
    nonredundancy_negative_num: 855610, nonredundancy_negatives: 853615
    corpus: 384913
    min frequency: 289
    
    python step9_subword_100.py \
        --func corpus \
        --outfile ../../../subword/extra_p_100/extra_p_100_corpus.txt
        
    python step9_subword_100.py \
        --func learn_bpe \
        --num_symbols 20000 \
        --infile ../../../subword/extra_p_100/extra_p_100_corpus.txt \
        --outfile ../../../subword/extra_p_100/extra_p_100_codes_20000.txt \
        --verbose
    
    python step9_subword_100.py  \
        --func tokenize  \
        --seq IPKIDNPEFASQYRPISCCNIFYKCISKMFCSRLKAVVLHLVAENQAAFVQGSQARGGAMDRITTTTRKFE \
        --codes_file ../../../subword/extra_p_100/extra_p_100_codes_20000.txt 
    
    python step9_subword_100.py  \
        --func apply_bpe  \
        --infile ../../../subword/extra_p_100/extra_p_100_corpus.txt  \
        --codes_file ../../../subword/extra_p_100/extra_p_100_codes_20000.txt \
        --outfile ../../../subword/extra_p_100/extra_p_100_corpus_token_20000.txt
    
    python step9_subword_100.py  \
        --func get_vocab  \
        --infile ../../../subword/extra_p_100/extra_p_100_corpus_token_20000.txt  \
        --outfile ../../../subword/extra_p_100/extra_p_100_subword_vocab_20000_ori.txt
    
    python step9_subword_100.py  \
        --func subword_vocab_2_token_vocab  \
        --infile ../../../subword/extra_p_100/extra_p_100_subword_vocab_20000_ori.txt  \
        --outfile ../../../vocab/extra_p_100/extra_p_100_subword_vocab_20000.txt
    '''