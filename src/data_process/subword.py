#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/15 13:56
@project: LucaPCycle
@file: subword
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
sys.path.append("../../src")
try:
    from file_operator import fasta_reader, csv_reader, txt_reader, txt_writer
except ImportError as e:
    from src.file_operator import fasta_reader, csv_reader, txt_reader, txt_writer


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


def generate_corpus(input_filepath, save_filepath, positive_label="Positive", other_rate=0.2):
    '''
    fasta to sequence corpus
    :param input_filepath:
    :param save_filepath:
    :return:
    '''
    selected_seqs = []
    if ".csv" in input_filepath:
        dataset = {}
        for row in csv_reader(input_filepath, header_filter=True, header=True):
            seq_id, seq_type, seq, label = row[0].strip(), row[1], row[2].strip().upper(), row[3]
            if label not in dataset:
                dataset[label] = []
            dataset[label].append(seq)
        for item in dataset.items():
            if item[0] == positive_label:
                selected_seqs.extend(item[1])
                print("label: %s, size: %d, selected size: %d" % (item[0], len(item[1]), len(item[1])))
            else:
                tmp = item[1]
                for _ in range(10):
                    random.shuffle(tmp)
                selected_size = int(len(tmp) * other_rate)
                selected_seqs.extend(tmp[:selected_size])
                print("label: %s, size: %d, selected size: %d" % (item[0], len(item[1]), selected_size))

    else:
        for row in fasta_reader(input_filepath):
            selected_seqs.append(row[1].strip().upper())
    with open(save_filepath, "w") as wfp:
        for seq in selected_seqs:
            wfp.write(seq + "\n")


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
            print("Warning: ouput dir %s not exists, created!" % dir_path)
            os.makedirs(dir_path)
    if args.func == "corpus":
        generate_corpus(args.infile, args.outfile)
    elif args.func == "learn_bpe":
        if ".fa" in args.infile:
            # transform fasta to corpus
            print('fasta convect to corpus txt')
            save_path = os.path.join(os.path.dirname(args.infile), ".".join(os.path.basename(args.infile).split(".")[0: -1]) + ".txt")
            if os.path.exists(save_path):
                raise Exception("Save path :%s exsits!" % save_path)
            generate_corpus(args.infile, save_path)
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
    label: Positive, size: 14322, selected size: 14322
    label: intra_P, size: 36661, selected size: 7332
    label: other, size: 63606, selected size: 12721
    label: uniprot_N, size: 579092, selected size: 115818
    label: N, size: 8447, selected size: 1689
    label: S, size: 63525, selected size: 12705
    min frequency: 115
    python subword.py \
        --func corpus \
        --infile ../../data/extra_p_50/cold_spring_sample_50.csv \
        --outfile ../../subword/extra_p_50/all_sequence_extra_p_50.txt
        
    python subword.py \
        --func learn_bpe \
        --num_symbols 20000 \
        --infile ../../subword/extra_p_50/all_sequence_extra_p_50.txt \
        --outfile ../../subword/extra_p_50/extra_p_50_codes_20000.txt \
        --verbose
    
    python subword.py  \
        --func tokenize  \
        --seq IPKIDNPEFASQYRPISCCNIFYKCISKMFCSRLKAVVLHLVAENQAAFVQGSQARGGAMDRITTTTRKFE \
        --codes_file ../../subword/extra_p_50/extra_p_50_codes_20000.txt 
    
    python subword.py  \
        --func apply_bpe  \
        --infile ../../subword/extra_p_50/all_sequence_extra_p_50.txt  \
        --codes_file ../../subword/extra_p_50/extra_p_50_codes_20000.txt \
        --outfile ../../subword/extra_p_50/all_sequence_token_20000.txt
    
    python subword.py  \
        --func get_vocab  \
        --infile ../../subword/extra_p_50/all_sequence_token_20000.txt  \
        --outfile ../../subword/extra_p_50/extra_p_50_subword_vocab_20000_ori.txt
    
    python subword.py  \
        --func subword_vocab_2_token_vocab  \
        --infile ../../subword/extra_p_50/extra_p_50_subword_vocab_20000_ori.txt  \
        --outfile ../../vocab/extra_p_50/extra_p_50_subword_vocab_20000.txt
        
          
    '''