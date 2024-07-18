#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/6/22 20:28
@project: LucaPCycle
@file: inference
@desc: inference one sample from input
'''
import os, sys, json, codecs
from subword_nmt.apply_bpe import BPE
from transformers.models.bert.tokenization_bert import BertTokenizer
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from common.alphabet import Alphabet
    from common.multi_label_metrics import *
    from lucaprot.models.lucaprot import LucaProt
    from utils import set_seed, plot_bins, load_trained_model, download_trained_checkpoint_lucapcycle
    from file_operator import csv_reader
    from llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from encoder import Encoder
    from batch_converter import BatchConverter
except ImportError:
    from src.common.alphabet import Alphabet
    from src.common.model_config import *
    from src.common.multi_label_metrics import *
    from src.lucaprot.models.lucaprot import LucaProt
    from src.utils import set_seed, plot_bins, load_trained_model, download_trained_checkpoint_lucapcycle
    from src.file_operator import csv_reader
    from src.llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter


def load_label_code_2_name(args, filename):
    '''
    load the mapping between the label name and label code
    :param args:
    :param filename:
    :return:
    '''
    label_code_2_name = {}
    filename = "../dataset/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, filename)
    if filename and os.path.exists(filename):
        with open(filename, "r") as rfp:
            for line in rfp:
                strs = line.strip().split("###")
                label_code_2_name[strs[0]] = strs[1]
    return label_code_2_name


def load_args(log_dir):
    '''
    load model running args
    :param log_dir:
    :return: config
    '''
    print("log dir: ", log_dir)
    log_filepath = os.path.join(log_dir, "logs.txt")
    if not os.path.exists(log_filepath):
        raise Exception("%s not exists" % log_filepath)
    with open(log_filepath, "r") as rfp:
        for line in rfp:
            if line.startswith("{"):
                obj = json.loads(line.strip())
                return obj
    return {}


def load_model(args, model_dir):
    '''
    create tokenizer, model config, model
    :param args:
    :return:
    '''
    # load labels
    label_filepath = args.label_filepath
    label_id_2_name = {}
    label_name_2_id = {}
    with open(label_filepath, "r") as fp:
        for line in fp:
            if line.strip() == "label":
                continue
            label_name = line.strip()
            label_id_2_name[len(label_id_2_name)] = label_name
            label_name_2_id[label_name] = len(label_name_2_id)
    print("-----------label_id_2_name:------------")
    if len(label_id_2_name) < 20:
        print(label_id_2_name)
    print("label size: ", len(label_id_2_name))
    model_config = LucaConfig.from_json_file(os.path.join(model_dir, "config.json"))

    seq_subword = None
    if args.input_type in ["seq", "seq_matrix", "seq_vector"]:
        if args.seq_subword:
            seq_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "sequence"), do_lower_case=args.do_lower_case)
            bpe_codes_prot = codecs.open(args.codes_file)
            seq_subword = BPE(bpe_codes_prot, merges=-1, separator='')
        else:
            seq_tokenizer = Alphabet.from_predefined(args.seq_vocab_path)
    else:
        seq_tokenizer = None

    # model class
    if args.model_type == "lucaprot":
        model_class = LucaProt
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)

    if model_dir and os.path.exists(model_dir):
        model = load_trained_model(model_config, args, model_class, model_dir)
    else:
        model = model_class(model_config, args)

    args.device = torch.device(args.device)
    model.to(args.device)
    model.eval()
    return model_config, seq_subword, seq_tokenizer, model, label_id_2_name, label_name_2_id


def predict_probs(model, batch_input):
    '''
    predict probs
    :param model:
    :param batch_input: a batch samples
    :return:
    '''
    if torch.cuda.is_available():
        probs = model(**batch_input)[1].detach().cpu().numpy()
    else:
        probs = model(**batch_input)[1].detach().numpy()
    return probs


def predict_binary_class(label_id_2_name, model, ori_input, batch_input, threshold=0.5):
    '''
    predict binary-class for a batch sample
    :param label_id_2_name:
    :param model:
    :param ori_input:
    :param batch_input: a batch samples
    :param threshold
    :return:
    '''
    probs = predict_probs(model, batch_input)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(ori_input):
        cur_res = info + [float(probs[idx][0]), label_id_2_name[preds[idx]]]
        res.append(cur_res)
    return res


def predict_multi_class(label_id_2_name, model, ori_input, batch_input, threshold=None):
    '''
    predict multi-class for a batch sample
    :param label_id_2_name:
    :param model:
    :param ori_input:
    :param batch_input: a batch samples
    :return:
    '''
    probs = predict_probs(model, batch_input)
    # print("probs dim: ", probs.ndim)
    preds = np.argmax(probs, axis=-1)
    res = []
    for idx, info in enumerate(ori_input):
        cur_res = info + [float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
        res.append(cur_res)
    return res


def predict_multi_label(label_id_2_name, model, ori_input, batch_input, threshold=0.5):
    '''
    predict multi-labels for a batch sample
    :param label_id_2_name:
    :param model:
    :param ori_input:
    :param batch_input: a batch samples
    :return:
    '''
    probs = predict_probs(model, batch_input)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= threshold).astype(int))
    res = []
    for idx, info in enumerate(ori_input):
        cur_res = info + [[float(probs[idx][label_index]) for label_index in preds[idx]], [label_id_2_name[label_index] for label_index in preds[idx]]]
        res.append(cur_res)
    return res


def load_environment(args):
    download_trained_checkpoint_lucapcycle(model_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = "../models/%s/%s/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, args.model_type, args.time_str, args.step if args.step == "best" else "checkpoint-{}".format(args.step))
    config_dir = "../logs/%s/%s/%s/%s/%s" % (args.dataset_name, args.dataset_type, args.task_type, args.model_type,  args.time_str)
    # Step1: loading the model configuration
    config = load_args(config_dir)
    print("-" * 25 + "config:" + "-" * 25)
    print(config)
    if config:
        args.dataset_name = config["dataset_name"]
        args.dataset_type = config["dataset_type"]
        args.task_type = config["task_type"]
        args.model_type = config["model_type"]
        args.input_type = config["input_type"]
        args.seq_subword = config["seq_subword"]
        args.codes_file = config["codes_file"]
        args.input_mode = config["input_mode"]
        args.seq_subword = config["seq_subword"]
        args.label_filepath = config["label_filepath"]
        args.codes_file = config["codes_file"]
        args.input_mode = config["input_mode"]
        args.label_type = config["label_type"]
        if not os.path.exists(args.label_filepath):
            args.label_filepath = os.path.join(config_dir, "label.txt")
        if args.batch_size is None and config["per_gpu_eval_batch_size"]:
            args.batch_size = config["per_gpu_eval_batch_size"]
        args.output_dir = config["output_dir"]
        args.config_path = config["config_path"]
        args.seq_vocab_path = config["seq_vocab_path"]
        args.seq_pooling_type = config["seq_pooling_type"]
        args.matrix_pooling_type = config["matrix_pooling_type"]
        args.matrix_encoder = config["matrix_encoder"]
        args.matrix_encoder_act = config["matrix_encoder_act"]
        args.fusion_type = config["fusion_type"]
        args.do_lower_case = config["do_lower_case"]
        args.sigmoid = config["sigmoid"]
        args.loss_type = config["loss_type"]
        args.output_mode = config["output_mode"]
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        args.seq_max_length = config["seq_max_length"]
        if args.seq_max_length and args.seq_max_length > args.truncation_seq_length:
            args.seq_max_length = args.truncation_seq_length
        args.no_position_embeddings = config["no_position_embeddings"]
        args.no_token_type_embeddings = config["no_token_type_embeddings"]
        args.embedding_input_size = config["embedding_input_size"]
        args.matrix_max_length = config["matrix_max_length"]
        if args.truncation_seq_length and args.matrix_max_length > args.truncation_seq_length:
            args.matrix_max_length = args.truncation_seq_length
        args.trunc_type = config["trunc_type"]
        args.llm_dir = config["llm_dir"]
        args.llm_type = config["llm_type"]
        args.llm_version = config["llm_version"]
        args.llm_task_level = config["llm_task_level"]
        args.llm_time_str = config["llm_time_str"]
        args.llm_step = config["llm_step"]
        args.llm_dirpath = config["llm_dirpath"]
        if args.task_type in ["multi-label", "multi_label", "binary-class", "binary_class"]:
            args.sigmoid = True
        args.llm_dirpath = config["llm_dirpath"]
        args.not_prepend_bos = config["not_prepend_bos"]
        args.not_append_eos = config["not_append_eos"]

    print("-" * 25 + "args:" + "-" * 25)
    print(args.__dict__.items())
    print("-" * 25 + "model_dir list:" + "-" * 25)
    print(os.listdir(model_dir))

    # Step2: loading the tokenizer and model
    model_config, seq_subword, seq_tokenizer, model, label_id_2_name, label_name_2_id = load_model(args, model_dir)
    encoder_config = {
        "llm_type": args.llm_type,
        "llm_dirpath": args.llm_dirpath,
        "input_type": args.input_type,
        "trunc_type": args.trunc_type,
        "seq_max_length": args.seq_max_length,
        "vector_dirpath": args.vector_dirpath,
        "matrix_dirpath": args.matrix_dirpath,
        "local_rank": -1
    }
    encoder = Encoder(**encoder_config)

    # encoding
    # luca独特的batch转换器
    batch_converter = BatchConverter(
        task_level_type=args.task_level_type,
        label_size=args.label_size,
        output_mode=args.output_mode,
        seq_subword=seq_subword,
        seq_tokenizer=seq_tokenizer,
        no_position_embeddings=model_config.no_position_embeddings,
        no_token_type_embeddings=model_config.no_token_type_embeddings,
        truncation_seq_length=model_config.seq_max_length,
        truncation_matrix_length=model_config.matrix_max_length,
        ignore_index=model_config.ignore_index,
        prepend_bos=not args.not_prepend_bos,
        append_eos=not args.not_append_eos
    )

    predict_func = None
    if args.task_type in ["multi-label", "multi_label"]:
        predict_func = predict_multi_label
    elif args.task_type in ["binary-class", "binary_class"]:
        predict_func = predict_binary_class
    elif args.task_type in ["multi-class", "multi_class"]:
        predict_func = predict_multi_class
    else:
        raise Exception("Not Support Task Type: %s" % args.task_type)

    return args, model_config, seq_subword, seq_tokenizer, model, label_id_2_name, label_name_2_id, encoder, batch_converter, predict_func