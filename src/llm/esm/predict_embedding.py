#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/3/20 13:23
@project: LucaPCycle
@file: predict_embedding
@desc: inference the embedding by ESM2
'''

import os
import sys
import esm
import torch
import random, argparse
from timeit import default_timer as timer
from esm import BatchConverter, pretrained
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader
    from utils import clean_seq, available_gpu_id, calc_emb_filename_by_seq_id
    from biotoolbox.structure_file_reader import *
    from biotoolbox.contact_map_builder import *
    from biotoolbox.contact_map_generator import *
except ImportError:
    from src.file_operator import fasta_reader, csv_reader
    from src.utils import clean_seq, available_gpu_id, calc_emb_filename_by_seq_id
    from src.biotoolbox.structure_file_reader import *
    from src.biotoolbox.contact_map_builder import *
    from src.biotoolbox.contact_map_generator import *


def enable_cpu_offloading(model):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:%d" % (7000 + random.randint(0, 1000)), world_size=1, rank=0
    )
    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def predict_pdb(sample, trunc_type, num_recycles=4, truncation_seq_length=4096, chunk_size=64, cpu_type="cpu-offload"):
    '''
    use sequence to predict protein 3D-structure
    :param sample:
    :param trunc_type:
    :param num_recycles:
    :param truncation_seq_length:
    :param chunk_size:
    :param cpu_type:
    :return: pdb, mean_plddt, ptm, processed_seq_len
    '''
    assert cpu_type is None or cpu_type in ["cpu-offload", "cpu-only"]
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model.set_chunk_size(chunk_size)
    if cpu_type == "cpu_only":
        model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
    elif cpu_type == "cpu_offload":
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()
    start = timer()
    protein_id, protein_seq = sample[0], sample[1]
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    cur_seq_len = len(protein_seq)
    processed_seq = protein_seq[:truncation_seq_length] if cur_seq_len > truncation_seq_length else protein_seq
    with torch.no_grad():
        try:
            output = model.infer([processed_seq], num_recycles=num_recycles)
            output = {key: value.cpu() for key, value in output.items()}
            mean_plddt = output["mean_plddt"][0]
            ptm = output["ptm"][0]
            pdb = model.output_to_pdb(output)[0]
            use_time = timer() - start
            print("predict pdb use time: %f" % use_time)
            return pdb, mean_plddt, ptm, processed_seq
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
            else:
                print(e)
    return None, None, None, None


def calc_distance_maps(pdb_filepath, chain, sequence):
    """
    Use the pdb file (3d-structure) of the protein to construct the adjacent Euclidean space distance matrix (diagonalized) between amino acids (residues),
        including two ways C_alpha (alpha carbon) and C_beta (beta carbon)
    :param pdb_filepath:
    :param chain: specified chain name
    :param sequence: the amino acids sequence
    :return: contact map
    """
    if ".cif" in pdb_filepath:
        ca = {}
        ca[chain] = {}
        ca[chain]['contact-map'] = ContactMap(pdb_filepath, None, chain=chain, c_atom_type="CA")
        cb = {}
        cb[chain] = {}
        cb[chain]['contact-map'] = ContactMap(pdb_filepath, None, chain=chain, c_atom_type="CB")
        return ca, cb
    else:
        if os.path.exists(pdb_filepath): # from file
            pdb_handle = open(pdb_filepath, 'r')
            pdb_content = pdb_handle.read()
        else: # input is pdb content
            pdb_handle = None
            pdb_content = pdb_filepath
        structure_container = build_structure_container_for_pdb(pdb_content, chain).with_seqres(sequence)
        # structure_container.chains = {chain: structure_container.chains[chain]}

        mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
        ca = mapper.generate_map_for_pdb(structure_container)
        cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
        if pdb_handle:
            pdb_handle.close()

        return ca.chains, cb.chains


global_model, global_alphabet, global_version, global_layer_size = None, None, None, None


def complete_embedding_matrix(seq_id, seq_type, seq, truncation_seq_length, init_emb, model_args, embedding_type):
    if init_emb is not None and model_args.embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        torch.cuda.empty_cache()
        ori_seq_len = len(seq)
        # 每次能处理这么长度
        # print("init_emb:", init_emb.shape)
        cur_segment_len = init_emb.shape[0]
        if model_args.matrix_add_special_token:
            first_emb = init_emb[1:cur_segment_len - 1]
        else:
            first_emb = init_emb
        if model_args.matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        # print("cur_segment_len: %d" % cur_segment_len)
        init_cur_segment_len = cur_segment_len
        segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
        if segment_num <= 1:
            return init_emb
        append_emb = None
        if model_args.embedding_complete_seg_overlap:
            sliding_window = init_cur_segment_len // 2
            print("Embedding Complete Seg Overlap: %r, ori seq len: %d, segment len: %d, init sliding windown: %d" % (model_args.embedding_complete_seg_overlap,
                                                                                                                      ori_seq_len, init_cur_segment_len, sliding_window))
            while True:
                print("updated window: %d" % sliding_window)
                try:
                    # 第一个已经处理，滑动窗口
                    if model_args.trunc_type == "right":
                        last_end = init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(init_cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                            seg_idx += 1
                            last_end = min(pos_idx + sliding_window, ori_seq_len)
                            seg_seq = seq[pos_idx - sliding_window:last_end]
                            print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                            seg_emb, seg_processed_seq_len = predict_embedding(sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                           trunc_type=model_args.trunc_type,
                                                                           embedding_type=embedding_type,
                                                                           repr_layers=[-1],
                                                                           truncation_seq_length=truncation_seq_length,
                                                                           device=model_args.device,
                                                                           version=model_args.llm_step,
                                                                           matrix_add_special_token=False
                                                                           )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[sliding_window:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[sliding_window:]), axis=0)
                        if last_end < ori_seq_len:
                            seg_idx += 1
                            remain = ori_seq_len - last_end
                            seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                            seg_emb, seg_processed_seq_len = predict_embedding(sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                           trunc_type=model_args.trunc_type,
                                                                           embedding_type=embedding_type,
                                                                           repr_layers=[-1],
                                                                           truncation_seq_length=truncation_seq_length,
                                                                           device=model_args.device,
                                                                           version=model_args.llm_step,
                                                                           matrix_add_special_token=False
                                                                           )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[-remain:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[-remain:]), axis=0)
                    else:
                        last_start = -init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(-init_cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                            seg_idx += 1
                            last_start = max(pos_idx - sliding_window, -ori_seq_len)
                            seg_seq = seq[last_start: pos_idx + sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding(sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                           trunc_type=model_args.trunc_type,
                                                                           embedding_type=embedding_type,
                                                                           repr_layers=[-1],
                                                                           truncation_seq_length=truncation_seq_length,
                                                                           device=model_args.device,
                                                                           version=model_args.llm_step,
                                                                           matrix_add_special_token=False
                                                                           )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:sliding_window]
                            else:
                                append_emb = np.concatenate((seg_emb[:sliding_window], append_emb), axis=0)
                        if last_start > -ori_seq_len:
                            seg_idx += 1
                            remain = last_start + ori_seq_len
                            seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding(sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                           trunc_type=model_args.trunc_type,
                                                                           embedding_type=embedding_type,
                                                                           repr_layers=[-1],
                                                                           truncation_seq_length=truncation_seq_length,
                                                                           device=model_args.device,
                                                                           version=model_args.llm_step,
                                                                           matrix_add_special_token=False
                                                                           )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:remain]
                            else:
                                append_emb = np.concatenate((seg_emb[:remain], append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change sliding window: %d -> %d" % (sliding_window, int(sliding_window * 0.95)))
                sliding_window = int(sliding_window * 0.95)
        else:
            while True:
                print("ori seq len: %d, segment len: %d" % (ori_seq_len, cur_segment_len))
                try:
                    # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
                    if model_args.trunc_type == "right":
                        begin_seq_idx = 0
                    else:
                        begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
                    for seg_idx in range(1, segment_num - 1):
                        seg_seq = seq[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                        # print("segment idx: %d, seg_seq(%d): %s" % (seg_idx, len(seg_seq), seg_seq))
                        print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                        seg_emb, seg_processed_seq_len = predict_embedding(
                            sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                            trunc_type=model_args.trunc_type,
                            embedding_type=embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device,
                            version=model_args.llm_step,
                            matrix_add_special_token=False
                        )

                        if append_emb is None:
                            append_emb = seg_emb
                        else:
                            '''
                            if model_args.trunc_type == "right":
                                append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                            else:
                                append_emb = np.concatenate((seg_emb, append_emb), axis=0)
                            '''
                            append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                    if model_args.trunc_type == "right":
                        # 处理最后一个
                        last_seg_seq = seq[-cur_segment_len:]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("last seg seq: %s" % last_seg_seq)
                        print("last seg seq len: %d, really len: %d" % (len(last_seg_seq), really_len))
                        last_seg_emb, last_seg_processed_seq_len = predict_embedding(
                            sample=[seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                            trunc_type=model_args.trunc_type,
                            embedding_type=embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device,
                            version=model_args.llm_step,
                            matrix_add_special_token=False
                        )
                        last_seg_emb = last_seg_emb[-really_len:, :]
                        append_emb = np.concatenate((append_emb, last_seg_emb), axis=0)
                    else:
                        # 处理第一个
                        first_seg_seq = seq[:cur_segment_len]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("first seg seq: %s" % first_seg_seq)
                        print("first seg seq len: %d, really len: %d" % (len(first_seg_seq), really_len))
                        first_seg_emb, first_seg_processed_seq_len = predict_embedding(sample=[seq_id + "_seg_0", seq_type, first_seg_seq],
                                                                                   trunc_type=model_args.trunc_type,
                                                                                   embedding_type=embedding_type,
                                                                                   repr_layers=[-1],
                                                                                   truncation_seq_length=truncation_seq_length,
                                                                                   device=model_args.device,
                                                                                   version=model_args.llm_step,
                                                                                   matrix_add_special_token=False
                                                                                   )
                        first_seg_emb = first_seg_emb[:really_len, :]
                        append_emb = np.concatenate((first_seg_emb, append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change segment len: %d -> %d, change seg num: %d -> %d" % (cur_segment_len, int(cur_segment_len * 0.95), segment_num, int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)))
                cur_segment_len = int(cur_segment_len * 0.95)
                segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)

            append_emb = append_emb[init_cur_segment_len - cur_segment_len:]
        if model_args.trunc_type == "right":
            complete_emb = np.concatenate((first_emb, append_emb), axis=0)
        else:
            complete_emb = np.concatenate((append_emb, first_emb), axis=0)
        print("seq len: %d, seq embedding matrix len: %d" % (ori_seq_len, complete_emb.shape[0] + (2 if model_args.matrix_add_special_token else 0)))
        print("-" * 50)
        assert complete_emb.shape[0] == ori_seq_len
        if model_args.matrix_add_special_token:
            complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
        init_emb = complete_emb
    return init_emb


def predict_embedding(sample,
                      trunc_type,
                      embedding_type,
                      repr_layers=[-1],
                      truncation_seq_length=4094,
                      device=None,
                      version="3B",
                      matrix_add_special_token=False):
    '''
    use sequence to predict protein embedding matrix or vector(bos)
    :param sample: [protein_id, protein_sequence]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :param device:
    :param version:
    :param matrix_add_special_token:
    :return: embedding, processed_seq_len
    '''
    global global_model, global_alphabet, global_version, global_layer_size
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        protein_id, protein_seq = sample[0], sample[2]
    else:
        protein_id, protein_seq = sample[0], sample[1]
    protein_seq = clean_seq(protein_id, protein_seq)
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    if global_model is None or global_alphabet is None or global_version is None or global_version != version or global_layer_size is None:
        if version == "15B":
            llm_name = "esm2_t48_15B_UR50D"
            global_layer_size = 48
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t48_15B_UR50D")
        elif version == "3B":
            llm_name = "esm2_t36_3B_UR50D"
            global_layer_size = 36
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
        elif version == "650M":
            llm_name = "esm2_t33_650M_UR50D"
            global_layer_size = 33
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        elif version == "150M":
            llm_name = "esm2_t30_150M_UR50D"
            global_layer_size = 30
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
        else:
            raise Exception("not support this version=%s" % version)
        print("LLM: %s, version: %s, layer_idx: %d, device: %s" % (llm_name, version, global_layer_size, str(device)))
        global_version = version
    '''
    if torch.cuda.is_available() and device is not None:
        global_model = global_model.to(device)
    elif torch.cuda.is_available():
        global_model = global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    '''
    if device is None:
        device = next(global_model.parameters()).device
    else:
        model_device = next(global_model.parameters()).device
        if device != model_device:
            global_model = global_model.to(device)
    # print("llm device:", device)
    assert all(-(global_model.num_layers + 1) <= i <= global_model.num_layers for i in repr_layers)
    repr_layers = [(i + global_model.num_layers + 1) % (global_model.num_layers + 1) for i in repr_layers]
    global_model.eval()

    converter = BatchConverter(global_alphabet, truncation_seq_length)
    protein_ids, raw_seqs, tokens = converter([[protein_id, protein_seq]])
    embeddings = {}
    with torch.no_grad():
        # if torch.cuda.is_available():
        tokens = tokens.to(device=device, non_blocking=True)
        try:
            out = global_model(tokens, repr_layers=repr_layers, return_contacts=False)
            truncate_len = min(truncation_seq_length, len(raw_seqs[0]))
            processed_seq_len = truncate_len + 2
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = out["representations"][global_layer_size].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                else:
                    embedding = out["representations"][global_layer_size].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out["representations"][global_layer_size].to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                embedding = out["contacts"][global_layer_size].to(device="cpu")[0, :, :].clone().numpy()
                embeddings["contacts"] = embedding
            if len(embeddings) > 1:
                return embeddings, processed_seq_len
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq_len
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1] if len(sample) == 2 else sample[2] )}.")
                print("Please reduce the 'truncation_seq_length'")
            return None, None


def get_args():
    parser = argparse.ArgumentParser(description='ESM/ESM2 Embedding')
    parser.add_argument("--llm_type", type=str, default="esm", choices=["esm"],
                        help="llm type")
    parser.add_argument("--llm_version", type=str, default="esm2", choices=["esm2"],
                        help="llm version")
    parser.add_argument("--llm_step", type=str, default="3B", choices=["15B", "3B", "650M", "150M"],
                        help="llm step")
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=["matrix", "vector", "contact"],
                        help="llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"],
                        help="llm trunc type of seq.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094,
                        help="truncation seq length.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="input filepath(format: fasta or csv)")
    parser.add_argument("--seq", type=str, default=None,
                        help="the input seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"],
                        help="seq type")
    parser.add_argument("--save_path", type=str, default=None,
                        help="embedding file save path")
    parser.add_argument("--id_idx", type=int, default=None,
                        help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None,
                        help="seq col idx(0 start)")
    parser.add_argument("--embedding_complete",  action="store_true",
                        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")
    parser.add_argument("--embedding_complete_seg_overlap",  action="store_true",
                        help="segment overlap")
    parser.add_argument("--matrix_add_special_token", action="store_true",
                        help="whether to add special token embedding in seq representation matrix")
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="gpu idx(-1 for CPU).")
    input_args = parser.parse_args()
    return input_args


def main(args):
    if args.gpu_id >= 0:
        gpu_id = args.gpu_id
    else:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        print("gpu_id: ", gpu_id)
    """
    if gpu_id is None or gpu_id == -1:
        args.device = None
    else:
        args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    """
    args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # global_model.to(args.device)
    assert args.input_file is not None or args.seq is not None
    print("input seq type: %s" % args.seq_type)
    print("args device: %s" % args.device)
    embedding_type = args.embedding_type
    save_path = args.save_path
    seq_type = args.seq_type
    print("emb save dir: %s" % save_path)
    if seq_type not in ["gene", "prot"]:
        seq_type = "prot"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.input_file:
        done = 0
        file_reader = fasta_reader
        if args.input_file.endswith(".csv"):
            file_reader = csv_reader
        for row in file_reader(args.input_file):
            if args.id_idx is None or args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[args.id_idx].strip(), row[args.seq_idx].upper()
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                ori_seq_len = len(seq)
                truncation_seq_length = args.truncation_seq_length
                if args.embedding_complete:
                    truncation_seq_length = ori_seq_len
                else:
                    truncation_seq_length = min(ori_seq_len, truncation_seq_length)
                emb, processed_seq_len = predict_embedding(sample=[seq_id, seq_type, seq],
                                                           trunc_type=args.trunc_type,
                                                           embedding_type=embedding_type,
                                                           repr_layers=[-1],
                                                           truncation_seq_length=truncation_seq_length,
                                                           device=args.device,
                                                           version=args.llm_step,
                                                           matrix_add_special_token=args.matrix_add_special_token
                                                           )
                while emb is None:
                    print("%s embedding error, max_len from %d truncate to %d" % (seq_id, truncation_seq_length,
                                                                                  int(truncation_seq_length * 0.95)))
                    truncation_seq_length = int(truncation_seq_length * 0.95)
                    emb, processed_seq_len = predict_embedding(sample=[seq_id, seq_type, seq],
                                                           trunc_type=args.trunc_type,
                                                           embedding_type=embedding_type,
                                                           repr_layers=[-1],
                                                           truncation_seq_length=truncation_seq_length,
                                                           device=args.device,
                                                           matrix_add_special_token=args.matrix_add_special_token,
                                                           version=args.llm_step,
                                                           )
                    if emb is not None:
                        emb = complete_embedding_matrix(seq_id=seq_id, seq_type=seq_type, seq=seq, truncation_seq_length=truncation_seq_length, init_emb=emb, model_args=args, embedding_type=embedding_type)
                # print("seq_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(emb, embedding_filepath)
                torch.cuda.empty_cache()
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)

    elif args.seq:
        print("input seq length: %d" % len(args.seq))
        emb, processed_seq_len = predict_embedding(sample=[args.seq_id, seq_type, args.seq],
                                                   trunc_type=args.trunc_type,
                                                   embedding_type=embedding_type,
                                                   repr_layers=[-1],
                                                   truncation_seq_length=args.truncation_seq_length,
                                                   device=args.device,
                                                   version=args.llm_step,
                                                   matrix_add_special_token=args.matrix_add_special_token)
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        print(emb.shape)


if __name__ == "__main__":
    args = get_args()
    main(args)