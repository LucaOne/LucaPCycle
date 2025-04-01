#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/6/22 20:28
@project: LucaPCycle
@file: inference_one
@desc: inference one sample from input
'''
import argparse
import os, sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from inference import load_environment
except ImportError:
    from src.inference import load_environment


def main():
    parser = argparse.ArgumentParser(description="Prediction")
    # the protein LLM exists path
    parser.add_argument("--torch_hub_dir", default=None, type=str,
                        help="set the torch hub dir path for saving pretrained model(default: ~/.cache/torch/hub)")
    parser.add_argument("--seq_id_a", default=None, type=str, required=True, help="the seq id of seq-a")
    parser.add_argument("--seq_type_a", default=None, type=str, required=True, help="the seq type of seq-a")
    parser.add_argument("--seq_a", default=None, type=str, required=True, help="the sequence")
    parser.add_argument("--seq_id_b", default=None, type=str, required=True, help="the seq id of seq-b")
    parser.add_argument("--seq_type_b", default=None, type=str, required=True, help="the seq type of seq-b")
    parser.add_argument("--seq_b", default=None, type=str, required=True, help="the sequence")
    parser.add_argument("--seq_id", default=None, type=str, required=True, help="the seq id")
    parser.add_argument("--seq_type", default=None, type=str, required=True, help="the seq type")
    parser.add_argument("--seq", default=None, type=str, required=True, help="the sequence")
    parser.add_argument("--truncation_seq_length", default=4096, type=int, required=True, help="truncation seq length")
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
    args, model_config, seq_subword, seq_tokenizer, model, label_id_2_name, label_name_2_id, \
    encoder, batch_converter, predict_func = load_environment(args)
    if args.input_mode == "single":
        ori_input = [[args.seq_id, args.seq_type, args.seq]]
        record = encoder.encode_single(args.seq_id, args.seq_type, args.seq)
    else:
        ori_input = [[args.seq_id_a, args.seq_id_b, args.seq_type_a, args.seq_type_b, args.seq_a, args.seq_b]]
        record = encoder.encode_pair(args.seq_id_a, args.seq_id_b, args.seq_type_a, args.seq_type_b, args.seq_a, args.seq_b)
    batch_input = batch_converter([record])
    # Step 3: prediction
    res = predict_func(args, label_id_2_name, model, ori_input, batch_input, threshold=args.threshold)
    print("res:")
    print(res[0])



