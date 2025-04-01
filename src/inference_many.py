#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/6/22 20:28
@project: LucaPCycle
@file: inference_many
@desc: inference many samples from input
'''
import argparse
import csv
import os.path
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from inference import load_environment
    from file_operator import csv_reader, fasta_reader, csv_writer
except ImportError:
    from src.inference import load_environment
    from src.file_operator import csv_reader, fasta_reader, csv_writer


def main():
    parser = argparse.ArgumentParser(description="Prediction")
    # the protein LLM exists path
    parser.add_argument("--torch_hub_dir", default=None, type=str,
                        help="set the torch hub dir path for saving pretrained model(default: ~/.cache/torch/hub)")
    parser.add_argument("--data_path", default=None, type=str, required=True, help="the data filepath")
    parser.add_argument("--seq_type", default=None, type=str, required=True, help="the seq type")
    parser.add_argument("--save_path", default=None, type=str, required=True, help="the results saved filepath")
    parser.add_argument("--truncation_seq_length", default=None, type=int, required=True, help="truncation seq length")
    parser.add_argument("--batch_size", default=8, type=int, required=True, help="batch size")
    parser.add_argument("--emb_dir", default=None, type=str, help="the structural embedding save dir. default: None")
    parser.add_argument("--dataset_name", default="extra_p", type=str, required=True, help="the dataset name for model building.")
    parser.add_argument("--dataset_type", default="protein", type=str, required=True, help="the dataset type for model building.")
    parser.add_argument("--task_type", default=None, type=str, required=True, choices=["multi_label", "multi_class", "binary_class"], help="the task type for model building.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="model type.")
    parser.add_argument("--time_str", default=None, type=str, required=True, help="the running time string(yyyymmddHimiss) of model building.")
    parser.add_argument("--step", default=None, type=str, required=True, help="the training global step of model finalization.")
    parser.add_argument("--threshold",  default=0.5, type=float, help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, default: 0.5.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    if args.torch_hub_dir is not None:
        if not os.path.exists(args.torch_hub_dir):
            os.makedirs(args.torch_hub_dir)
        os.environ['TORCH_HOME'] = args.torch_hub_dir
    args, model_config, seq_subword, seq_tokenizer, model, label_id_2_name, \
    label_name_2_id, encoder, batch_converter, predict_func = load_environment(args)
    assert args.data_path is not None and os.path.exists(args.data_path)
    file_format = args.data_path.split(".")[-1]
    with open(args.save_path, "w") as wfp:
        writer = csv.writer(wfp)
        cur_ori_input = []
        cur_batch = []
        if file_format in ["fa", "fas", "fasta"]:
            if args.input_mode in ["pair"]:
                raise Exception("Pair Input Task not support fasta file.")
            writer.writerow(["seq_id", "seq_type", "seq", "prob", "label"])
            for row in fasta_reader(args.data_path):
                cur_ori_input.append([row[0], args.seq_type, row[1]])
                record = encoder.encode_single(row[0], args.seq_type, row[1])
                cur_batch.append(record)
                if len(cur_batch) == args.batch_size:
                    cur_batch_input = batch_converter(cur_batch)
                    res = predict_func(args, label_id_2_name, model, cur_ori_input, cur_batch, threshold=args.threshold)
                    for item in res:
                        writer.writerow(item)
                    cur_ori_input = []
                    cur_batch = []
        else:
            for row in csv_reader(args.data_path, header_filter=True, header=True):
                if args.input_mode == "single":
                    cur_ori_input.append([row[0], row[1], row[2]])
                    record = encoder.encode_single(row[0], row[1], row[2])
                    cur_batch.append(record)
                else:
                    cur_ori_input.append([row[0], row[1], row[2], row[3], row[4], row[5]])
                    record = encoder.encode_pair(row[0], row[1], row[2], row[3], row[4], row[5])
                    cur_batch.append(record)
                if len(cur_batch) == args.batch_size:
                    cur_batch_input = batch_converter(cur_batch)
                    res = predict_func(args, label_id_2_name, model, cur_ori_input, cur_batch_input, threshold=args.threshold)
                    for item in res:
                        writer.writerow(item)
                    cur_ori_input = []
                    cur_batch = []
        if len(cur_batch) > 0:
            cur_batch_input = batch_converter(cur_batch)
            res = predict_func(args, label_id_2_name, model, cur_ori_input, cur_batch, threshold=args.threshold)
            for item in res:
                writer.writerow(item)



