#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/10 20:18
@project: LucaPCycle
@file: prediction
@desc: predict one sample from file

'''
import csv
import json
import os, sys
import torch
import codecs
import time, shutil
import numpy as np
import argparse
from collections import OrderedDict
from subword_nmt.apply_bpe import BPE
from transformers import BertConfig, BertTokenizer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device
    from common.multi_label_metrics import relevant_indexes
    from common.alphabet import Alphabet
    from encoder import Encoder
    from batch_converter import BatchConverter
    from lucaprot.models.lucaprot import LucaProt
    from utils import available_gpu_id, load_labels, download_trained_checkpoint_lucapcycle
    from file_operator import csv_reader, fasta_reader, csv_writer, tsv_reader
except ImportError:
    from src.utils import to_device
    from src.common.multi_label_metrics import relevant_indexes
    from src.common.alphabet import Alphabet
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.lucaprot.models.lucaprot import LucaProt
    from src.utils import available_gpu_id, load_labels, download_trained_checkpoint_lucapcycle
    from src.file_operator import csv_reader, fasta_reader, csv_writer, tsv_reader


def transform_one_sample_2_feature(
        device,
        input_mode,
        encoder,
        batch_convecter,
        row
):
    """
    transform the raw input to model's input
    :param device:
    :param input_mode:
    :param encoder:
    :param batch_convecter:
    :param row:
    :return:
    """
    batch_info = []
    # for pair
    if input_mode == "pair":
        seq_lens = []
        en = encoder.encode_pair(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            vector_filename_a=None,
            vector_filename_b=None,
            matrix_filename_a=None,
            matrix_filename_b=None,
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[1], row[4], row[5]])
        seq_lens.append([len(row[4]), len(row[5])])
    else:
        seq_lens = []
        en_list = []
        cur_seq = row[2]
        # token level task
        if batch_convecter.task_level_type not in ["seq_level", "seq-level"]:
            # too long to segment
            split_seqs = []
            max_len = 10240 - int(batch_convecter.seq_prepend_bos) - int(batch_convecter.seq_append_eos)
            while max_len < len(cur_seq):
                split_seqs.append(cur_seq[:max_len])
                seq_lens.append(max_len)
                cur_seq = cur_seq[max_len:]
            if len(cur_seq) > 0:
                split_seqs.append(cur_seq)
                seq_lens.append(len(cur_seq))

            for split_seq in split_seqs:
                en = encoder.encode_single(
                    row[0],
                    row[1],
                    split_seq,
                    vector_filename=None,
                    matrix_filename=None,
                    label=None
                )
                en_list.append(en)
        else:
            en = encoder.encode_single(
                row[0],
                row[1],
                row[2],
                vector_filename=None,
                matrix_filename=None,
                label=None

            )
            en_list = en
            seq_lens = len(row[2])
            if "matrix" in en and en["matrix"] is not None:
                seq_lens = min(seq_lens, en["matrix"].shape[0]
                               - int(batch_convecter.seq_prepend_bos)
                               - int(batch_convecter.seq_append_eos))
        batch_info.append([row[0], row[2]])

    batch = [en_list]
    if isinstance(batch[0], list):
        batch_features = []
        for cur_batch in batch[0]:
            cur_batch_features = batch_convecter([cur_batch])
            cur_batch_features, cur_sample_num = to_device(device, cur_batch_features)
            batch_features.append(cur_batch_features)

    else:
        batch_features = batch_convecter(batch)
        batch_features, cur_sample_num = to_device(device, batch_features)
    return batch_info, batch_features, [seq_lens]


def predict_probs(
        args,
        encoder,
        batch_convecter,
        model,
        row
):
    """
    predict the prob
    :param args:
    :param encoder:
    :param batch_convecter:
    :param model:
    :param row:
    :return:
    """
    model.to(torch.device("cpu"))
    batch_info, batch_features, seq_lens = transform_one_sample_2_feature(
        args.device,
        args.input_mode,
        encoder,
        batch_convecter,
        row
    )
    model.to(args.device)
    if isinstance(batch_features, list):
        probs = []
        for cur_batch_features in batch_features:
            cur_probs = model(**cur_batch_features)[1]
            if cur_probs.is_cuda:
                cur_probs = cur_probs.detach().cpu().numpy()
            else:
                cur_probs = cur_probs.detach().numpy()
            probs.append(cur_probs)
    else:
        probs = model(**batch_features)[1]
        if probs.is_cuda:
            probs = probs.detach().cpu().numpy()
        else:
            probs = probs.detach().numpy()
    return batch_info, probs, seq_lens


def predict_seq_level_binary_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    """
    predict the seq level binary-class classification task
    :param args:
    :param encoder:
    :param batch_convecter:
    :param label_id_2_name:
    :param model:
    :param row:
    :return:
    """
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[4], info[5], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_multi_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row,
        topk=5
):
    """
    predict the seq level multi-class classification task
    :param args:
    :param encoder:
    :param batch_convecter:
    :param label_id_2_name:
    :param model:
    :param row:
    :param topk:
    :return:
    """
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)

    if topk is not None and topk > 1:
        # print("topk: %d" % topk)
        preds = np.argmax(probs, axis=-1)
        preds_topk = np.argsort(probs, axis=-1)[:, ::-1][:, :topk]
        res = []
        for idx, info in enumerate(batch_info):
            cur_topk_probs = []
            cur_topk_labels = []
            for label_idx in preds_topk[idx]:
                cur_topk_probs.append(float(probs[idx][label_idx]))
                cur_topk_labels.append(label_id_2_name[label_idx])
            if args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res
    else:
        preds = np.argmax(probs, axis=-1)
        res = []
        for idx, info in enumerate(batch_info):
            if args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]]
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res


def predict_seq_level_multi_label(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    """
    predict the seq level multi-label classification task
    :param args:
    :param encoder:
    :param batch_convecter:
    :param label_id_2_name:
    :param model:
    :param row:
    :return:
    """
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                [float(probs[idx][label_index]) for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                [float(probs[idx][label_index]) for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_regression(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    """
    predict the seq level regression task
    :param args:
    :param encoder:
    :param batch_convecter:
    :param label_id_2_name:
    :param model:
    :param row:
    :return:
    """
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                float(probs[idx][0]),
                float(probs[idx][0])
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                float(probs[idx][0]),
                float(probs[idx][0])
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def load_tokenizer(
        args,
        model_dir,
        seq_tokenizer_class,
        struct_tokenizer_class
):
    seq_subword = None
    seq_tokenizer = None
    if not hasattr(args, "has_seq_encoder") or args.has_seq_encoder:
        if args.seq_subword:
            if os.path.exists(os.path.join(model_dir, "sequence")):
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "sequence"), do_lower_case=args.do_lower_case)
            else:
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "tokenizer"), do_lower_case=args.do_lower_case)
            bpe_codes = codecs.open(args.codes_file)
            seq_subword = BPE(bpe_codes, merges=-1, separator='')
        else:
            seq_subword = None
            seq_tokenizer = seq_tokenizer_class.from_predefined(args.seq_vocab_path)
            if args.not_prepend_bos:
                seq_tokenizer.prepend_bos = False
            if args.not_append_eos:
                seq_tokenizer.append_eos = False
    if hasattr(args, "has_struct_encoder") and args.has_struct_encoder and struct_tokenizer_class is not None:
        struct_tokenizer = struct_tokenizer_class.from_pretrained(os.path.join(model_dir, "struct"), do_lower_case=args.do_lower_case)
    else:
        struct_tokenizer = None
    return seq_subword, seq_tokenizer, struct_tokenizer


def load_trained_model(
        model_config,
        args,
        model_class,
        model_dirpath
):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(os.path.join(model_dirpath, "pytorch.pth"),
                                         map_location=torch.device("cpu"))
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        print("diff:")
        print(model_state_dict_keys.difference(new_state_dict.keys()))
        model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()
    return model


def load_model(
        args,
        model_name,
        model_dir
):
    # load tokenizer and model
    begin_time = time.time()
    device = torch.device(args.device)
    print("load model on cuda:", device)
    if model_name.lower() == "lucaprot":
        if args.seq_subword:
            config_class, seq_tokenizer_class, struct_tokenizer_class, model_class = BertConfig, BertTokenizer, BertTokenizer, LucaProt
        else:
            config_class, seq_tokenizer_class, struct_tokenizer_class, model_class = BertConfig, Alphabet, Alphabet, LucaProt
    else:
        raise Exception("Not support model_name=%s" % model_name)
    seq_subword, seq_tokenizer, struct_tokenizer = load_tokenizer(args, model_dir, seq_tokenizer_class, struct_tokenizer_class)
    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    model_config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))

    model = load_trained_model(model_config, args, model_class, model_dir)
    print("the time for loading model:", time.time() - begin_time)

    return model_config, seq_subword, seq_tokenizer, struct_tokenizer, model


def create_encoder_batch_convecter(
        lucapcycle_args,
        seq_subword,
        seq_tokenizer
):
    if hasattr(lucapcycle_args, "input_mode") and lucapcycle_args.input_mode == "pair":
        raise Exception("Not support the input_mode=%s" % lucapcycle_args.input_mode)
    encoder_config = {
        "llm_type": lucapcycle_args.llm_type,
        "llm_version": lucapcycle_args.llm_version,
        "llm_step": lucapcycle_args.llm_step,
        "llm_dirpath": lucapcycle_args.llm_dirpath,
        "input_type": lucapcycle_args.input_type,
        "trunc_type": lucapcycle_args.trunc_type,
        "seq_max_length": lucapcycle_args.seq_max_length,
        "prepend_bos": True,
        "append_eos": True,
        "vector_dirpath": lucapcycle_args.vector_dirpath,
        "matrix_dirpath": lucapcycle_args.matrix_dirpath,
        "local_rank": lucapcycle_args.gpu_id,
        "embedding_complete": lucapcycle_args.embedding_complete,
        "embedding_complete_seg_overlap": lucapcycle_args.embedding_complete_seg_overlap,
        "matrix_add_special_token": lucapcycle_args.matrix_add_special_token,
        "matrix_embedding_exists": lucapcycle_args.matrix_embedding_exists,
        "use_cpu": True if lucapcycle_args.gpu_id < 0 else False
    }
    print("-" * 15 + "encoder_config:" + "-" * 15)
    print(encoder_config)
    print("-" * 50)
    encoder = Encoder(**encoder_config)

    batch_converter = BatchConverter(
        input_type=lucapcycle_args.input_type if hasattr(lucapcycle_args, "input_type") else False,
        task_level_type=lucapcycle_args.task_level_type,
        label_size=lucapcycle_args.label_size,
        output_mode=lucapcycle_args.output_mode,
        seq_subword=seq_subword,
        seq_tokenizer=seq_tokenizer,
        no_position_embeddings=lucapcycle_args.no_position_embeddings,
        no_token_type_embeddings=lucapcycle_args.no_token_type_embeddings,
        truncation_seq_length=lucapcycle_args.truncation_seq_length if hasattr(lucapcycle_args, "truncation_seq_length") else lucapcycle_args.seq_max_length,
        truncation_matrix_length=lucapcycle_args.truncation_matrix_length if hasattr(lucapcycle_args, "truncation_matrix_length") else lucapcycle_args.matrix_max_length,
        trunc_type=lucapcycle_args.trunc_type if hasattr(lucapcycle_args, "trunc_type") else "right",
        ignore_index=lucapcycle_args.ignore_index,
        non_ignore=lucapcycle_args.non_ignore,
        padding_idx=0,
        unk_idx=1,
        cls_idx=2,
        eos_idx=3,
        mask_idx=4,
        seq_prepend_bos=not lucapcycle_args.not_seq_prepend_bos,
        seq_append_eos=not lucapcycle_args.not_seq_append_eos,
        matrix_prepend_bos=not lucapcycle_args.not_matrix_prepend_bos,
        matrix_append_eos=not lucapcycle_args.not_matrix_append_eos,
        matrix_add_special_token=lucapcycle_args.matrix_add_special_token if hasattr(lucapcycle_args, "matrix_add_special_token") else False,
    )
    return encoder, batch_converter


# global
global_model_config, global_seq_subword, global_seq_tokenizer, \
global_struct_tokenizer, global_lucabase_model = None, None, None, None, None


def run(
        sequences,
        truncation_seq_length,
        model_path,
        dataset_name,
        dataset_type,
        task_type,
        task_level_type,
        model_type,
        input_type,
        input_mode,
        time_str,
        step,
        gpu_id,
        threshold,
        topk,
        emb_dir,
        matrix_embedding_exists
):
    # step1: loading args
    global global_model_config, global_seq_subword, global_seq_tokenizer, global_struct_tokenizer, global_lucabase_model
    # download_trained_checkpoint_lucapcycle(model_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,
        time_str, step if step == "best" else "checkpoint-{}".format(step)
    )
    config_dir = "%s/logs/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type, time_str
    )

    lucapcycle_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    print("LucaPCycle Args:")
    print(lucapcycle_args.__dict__)
    print("*" * 50)
    # 选较大值
    lucapcycle_args.truncation_seq_length = lucapcycle_args.seq_max_length
    if lucapcycle_args.truncation_seq_length is None or lucapcycle_args.truncation_seq_length < truncation_seq_length:
        lucapcycle_args.truncation_seq_length = truncation_seq_length
    # 选较大值
    lucapcycle_args.truncation_matrix_length = lucapcycle_args.matrix_max_length
    if lucapcycle_args.truncation_matrix_length is None or lucapcycle_args.truncation_matrix_length < truncation_seq_length:
        lucapcycle_args.truncation_matrix_length = truncation_seq_length

    lucapcycle_args.matrix_embedding_exists = matrix_embedding_exists
    lucapcycle_args.emb_dir = emb_dir
    lucapcycle_args.vector_dirpath = lucapcycle_args.emb_dir if lucapcycle_args.emb_dir else None
    lucapcycle_args.matrix_dirpath = lucapcycle_args.emb_dir if lucapcycle_args.emb_dir else None

    lucapcycle_args.dataset_name = dataset_name
    lucapcycle_args.dataset_type = dataset_type
    lucapcycle_args.task_level_type = task_level_type
    lucapcycle_args.task_type = task_type
    lucapcycle_args.model_type = model_type
    lucapcycle_args.input_type = input_type
    lucapcycle_args.input_mode = input_mode
    lucapcycle_args.time_str = time_str
    lucapcycle_args.step = step
    lucapcycle_args.threshold = threshold
    lucapcycle_args.gpu_id = gpu_id

    if not hasattr(lucapcycle_args, "embedding_complete"):
        lucapcycle_args.embedding_complete = False

    if not hasattr(lucapcycle_args, "embedding_complete_seg_overlap"):
        lucapcycle_args.embedding_complete_seg_overlap = False

    if not hasattr(lucapcycle_args, "not_seq_prepend_bos"):
        lucapcycle_args.not_seq_prepend_bos = False

    if not hasattr(lucapcycle_args, "not_matrix_prepend_bos"):
        lucapcycle_args.not_matrix_prepend_bos = True

    if not hasattr(lucapcycle_args, "not_seq_append_eos"):
        lucapcycle_args.not_seq_append_eos = False

    if not hasattr(lucapcycle_args, "not_matrix_append_eos"):
        lucapcycle_args.not_matrix_append_eos = True

    if not hasattr(lucapcycle_args, "matrix_add_special_token"):
        lucapcycle_args.matrix_add_special_token = False

    if not hasattr(lucapcycle_args, "non_ignore"):
        lucapcycle_args.non_ignore = True
    if not hasattr(lucapcycle_args, "ignore_index"):
        lucapcycle_args.ignore_index = -100

    if lucapcycle_args.label_filepath:
        lucapcycle_args.label_filepath = lucapcycle_args.label_filepath.replace("../", "%s/" % model_path)
    if not os.path.exists(lucapcycle_args.label_filepath):
        lucapcycle_args.label_filepath = os.path.join(config_dir, "label.txt")

    if gpu_id is None or gpu_id < 0:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        lucapcycle_args.gpu_id = gpu_id
    print("gpu_id: %d" % gpu_id)
    print("*" * 50)
    lucapcycle_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")

    print("LucaPCycle Args:")
    print(lucapcycle_args.__dict__)
    print("*" * 50)

    # Step2: loading the tokenizer and model
    if global_lucabase_model is None or next(global_lucabase_model.parameters()).device != lucapcycle_args.device:
        global_lucabase_model = None
        model_config, seq_subword, seq_tokenizer, struct_tokenizer, lucabase_model = \
            load_model(
                args=lucapcycle_args,
                model_name=model_type,
                model_dir=model_dir
            )
        global_model_config = model_config
        global_seq_subword = seq_subword
        global_seq_tokenizer = seq_tokenizer
        global_struct_tokenizer = struct_tokenizer
        global_lucabase_model = lucabase_model
    else:
        model_config = global_model_config
        seq_subword = global_seq_subword
        seq_tokenizer = global_seq_tokenizer
        struct_tokenizer = global_struct_tokenizer
        lucabase_model = global_lucabase_model

    encoder, batch_convecter = create_encoder_batch_convecter(
        lucapcycle_args,
        seq_subword,
        seq_tokenizer
    )

    # embedding in advance
    if not matrix_embedding_exists and gpu_id > -1:
        # 先to cpu
        lucabase_model.to(torch.device("cpu"))
        assert lucapcycle_args.emb_dir is not None
        if not os.path.exists(lucapcycle_args.emb_dir):
            os.makedirs(lucapcycle_args.emb_dir)
        for item in sequences:
            seq_id = item[0]
            seq_type = item[1]
            seq = item[2]
            encoder.__get_embedding__(
                seq_id=seq_id,
                seq_type=seq_type,
                seq=seq,
                embedding_type="matrix" if "matrix" in input_type else "vector"
            )
        encoder.matrix_embedding_exists = True
        # embedding 完之后to device
        lucabase_model.to(lucapcycle_args.device)

    label_list = load_labels(lucapcycle_args.label_filepath)
    label_id_2_name = {idx: name for idx, name in enumerate(label_list)}

    # Step 3: prediction
    if lucapcycle_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_seq_level_binary_class
    elif lucapcycle_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_seq_level_multi_class
    elif lucapcycle_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_seq_level_multi_label
    elif lucapcycle_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["regression"]:
        predict_func = predict_seq_level_regression
    else:
        raise Exception("the task_type=%s or task_level_type=%s error" % (task_type, lucapcycle_args.task_level_type))

    predicted_results = []
    for item in sequences:
        seq_id = item[0]
        seq_type = item[1]
        seq = item[2]
        row = [seq_id, seq_type, seq]
        if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
            # print("task_level_type: %s, task_type: %s" % (task_level_type, task_type))
            cur_res = predict_func(
                lucapcycle_args,
                encoder,
                batch_convecter,
                label_id_2_name,
                lucabase_model,
                row,
                topk=topk
            )
            if topk is not None and topk > 1:
                predicted_results.append([seq_id, seq, cur_res[0][2], cur_res[0][3], cur_res[0][4], cur_res[0][5]])
            else:
                predicted_results.append([seq_id, seq, cur_res[0][2], cur_res[0][3]])
        else:
            cur_res = predict_func(
                lucapcycle_args,
                encoder,
                batch_convecter,
                label_id_2_name,
                lucabase_model,
                row
            )
            predicted_results.append([seq_id, seq, cur_res[0][2], cur_res[0][3]])
    # torch.cuda.empty_cache()
    # 删除embedding
    if os.path.exists(lucapcycle_args.emb_dir):
        shutil.rmtree(lucapcycle_args.emb_dir)
    return predicted_results


def run_args():
    parser = argparse.ArgumentParser(description="Prediction")
    # for one seq
    parser.add_argument("--seq_id", default=None, type=str,
                        help="the seq id")
    parser.add_argument("--seq", default=None, type=str,
                        help="the sequence")
    parser.add_argument("--seq_type", default=None, type=str, required=True,
                        choices=["gene", "prot"],
                        help="the input seq type.")

    # for many seqs
    parser.add_argument("--input_file", default=None, type=str,
                        help="the input file(fasta or csv format).")
    parser.add_argument("--seq_id_idx", default=None, type=int,
                        help="the seq id index for csv.")
    parser.add_argument("--seq_idx", default=None, type=int,
                        help="the seq index for csv.")
    # for results saved
    parser.add_argument("--save_path", default=None, type=str,
                        help="the result save dir path")

    # for embedding
    parser.add_argument("--matrix_embedding_exists", action="store_true",
                        help="the structural embedding is or not in advance. default: False")
    parser.add_argument("--emb_dir", default=None, type=str,
                        help="the structural embedding save dir in advance. default: None")
    parser.add_argument("--truncation_seq_length", default=4096, type=int,
                        help="the truncation seq length for LLM, default: 4096")

    # for the trained model checkpoint
    parser.add_argument("--model_path", default="..", type=str,
                        help="the model dir. default: ../")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="the dataset name for model building.")
    parser.add_argument("--dataset_type", default="protein", type=str,
                        help="the dataset type for model building. default: protein ")
    parser.add_argument("--task_type", default=None, type=str, required=True,
                        choices=["binary_class", "multi_class"],
                        help="the task type for model building.")
    parser.add_argument("--task_level_type", default="seq_level", type=str,
                        choices=["seq_level", "token_level"],
                        help="the task level type for model building. default: seq_level")
    parser.add_argument("--model_type", default="lucaprot", type=str,
                        choices=["lucaprot"],
                        help="the model type. default: lucaprot")
    parser.add_argument("--input_type", default="seq_matrix", type=str, choices=[
        "seq",
        "matrix",
        "vector",
        "seq_matrix",
        "seq_vector"
    ],
                        help="the model channels. default: seq_matrix")
    parser.add_argument("--input_mode", default=None, type=str, required=True,
                        choices=["single", "pair"],
                        help="input mode.")
    parser.add_argument("--time_str", default=None, type=str, required=True,
                        help="the running time string(yyyymmddHimiss) of trained model.")
    parser.add_argument("--step", default=None, type=str, required=True,
                        help="the global step of trained model.")

    # for running
    parser.add_argument("--topk", default=None, type=int,
                        help="the topk labels for multi-class, default: None")
    parser.add_argument("--threshold",  default=0.1, type=float,
                        help="the positive(>=threshold) or negative(<threshold), default: 0.1. "
                             "Small value leads to high recall, and large value to high precision")

    # for print
    parser.add_argument("--print_per_number", default=10000, type=int,
                        help="per num to print, default: 10000")
    parser.add_argument("--gpu_id", default=-1, type=int, help="the used gpu_id. default: -1(CPU)")
    input_args = parser.parse_args()
    return input_args


if __name__ == "__main__":
    args = run_args()
    print("-" * 25 + "Run Args" + "-" * 25)
    print(args.__dict__)
    print("-" * 50)
    if args.emb_dir is not None and args.input_file is not None:
        emb_base_name = os.path.basename(args.emb_dir)
        input_base_name = ".".join(os.path.basename(args.input_file).split(".")[:-1])
        if emb_base_name != input_base_name:
            args.emb_dir = os.path.join(args.emb_dir, input_base_name)
            print("updated emb_dir: %s" % args.emb_dir)

    assert args.seq is not None or (args.input_file is not None and os.path.exists(args.input_file))
    if args.input_file is not None and os.path.exists(args.input_file):
        file_suffix = os.path.basename(args.input_file).split(".")[-1]
        if file_suffix in ["fasta", "faa", "fas", "fa"] and args.seq_type is None:
            print("Input a fasta file, please set arg: --seq_type, value: gene or prot")
            sys.exit(-1)

        exists_ids = set()
        exists_res = []
        if os.path.exists(args.save_path):
            print("save_path=%s exists." % args.save_path)
            for row in csv_reader(args.save_path, header=True, header_filter=True):
                if len(row) < 4:
                    continue
                exists_ids.add(row[0])
                exists_res.append(row)
            print("exists records: %d" % len(exists_res))
            print("*" * 50)
        elif not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        with open(args.save_path, "w") as wfp:
            writer = csv.writer(wfp)
            if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                writer.writerow(["seq_id", "seq", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk])
            else:
                writer.writerow(["seq_id", "seq", "prob", "label"])
            for item in exists_res:
                writer.writerow(item)
            exists_res = []
            batch_data = []
            had_done = 0

            need_col_index = False
            if args.input_file.endswith(".csv"):
                file_reader = csv_reader
                need_col_index = True
            elif args.input_file.endswith(".tsv"):
                file_reader = tsv_reader
                need_col_index = True
            else:
                file_reader = fasta_reader
            if need_col_index:
                assert args.seq_id_idx is not None and args.seq_idx is not None
            else:
                # for fasta
                args.seq_id_idx = 0
                args.seq_idx = 1

            file_reader = csv_reader if args.input_file.endswith(".csv") else fasta_reader
            for row in file_reader(args.input_file):
                if row[args.seq_id_idx] in exists_ids:
                    continue
                batch_data.append([row[args.seq_id_idx], args.seq_type, row[args.seq_idx]])
                if len(batch_data) % args.print_per_number == 0:
                    batch_results = run(
                        batch_data,
                        args.truncation_seq_length,
                        args.model_path,
                        args.dataset_name,
                        args.dataset_type,
                        args.task_type,
                        args.task_level_type,
                        args.model_type,
                        args.input_type,
                        args.input_mode,
                        args.time_str,
                        args.step,
                        args.gpu_id,
                        args.threshold,
                        topk=args.topk,
                        emb_dir=args.emb_dir,
                        matrix_embedding_exists=args.matrix_embedding_exists
                    )
                    for item in batch_results:
                        writer.writerow(item)
                    had_done += len(batch_data)
                    print("done: %d, had_done: %d" % (len(batch_data), had_done))
                    batch_data = []
            if len(batch_data) > 0:
                batch_results = run(
                    batch_data,
                    args.truncation_seq_length,
                    args.model_path,
                    args.dataset_name,
                    args.dataset_type,
                    args.task_type,
                    args.task_level_type,
                    args.model_type,
                    args.input_type,
                    args.input_mode,
                    args.time_str,
                    args.step,
                    args.gpu_id,
                    args.threshold,
                    topk=args.topk,
                    emb_dir=args.emb_dir,
                    matrix_embedding_exists=args.matrix_embedding_exists
                )
                for item in batch_results:
                    writer.writerow(item)
                had_done += len(batch_data)
                batch_data = []
            print("over, done: %d" % had_done)

    elif args.seq is not None:
        if args.seq_id is None:
            args.seq_id = "unknown_id"
        if args.seq_type is None:
            print("Please set arg: --seq_type, value: gene or prot")
        data = [[args.seq_id, args.seq_type, args.seq]]
        results = run(
            data,
            args.truncation_seq_length,
            args.model_path,
            args.dataset_name,
            args.dataset_type,
            args.task_type,
            args.task_level_type,
            args.model_type,
            args.input_type,
            args.input_mode,
            args.time_str,
            args.step,
            args.gpu_id,
            args.threshold,
            topk=args.topk,
            emb_dir=args.emb_dir,
            matrix_embedding_exists=args.matrix_embedding_exists
        )
        print("Predicted result:")
        print("seq_id=%s" % args.seq_id)
        print("seq=%s" % args.seq)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
        print("*" * 50)
    else:
        raise Exception("input error, usage: --hep")

