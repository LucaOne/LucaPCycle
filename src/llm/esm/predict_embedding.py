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
    from file_operator import fasta_reader
    from utils import clean_seq, available_gpu_id
    from biotoolbox.structure_file_reader import *
    from biotoolbox.contact_map_builder import *
    from biotoolbox.contact_map_generator import *
except ImportError:
    from src.file_operator import fasta_reader
    from src.utils import clean_seq, available_gpu_id
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


global_model, global_alphabet, global_version = None, None, None


def predict_embedding(sample, trunc_type, embedding_type, repr_layers=[-1], truncation_seq_length=4094, device=None, version="3B", matrix_add_special_token=False):
    '''
    use sequence to predict protein embedding matrix or vector(bos)
    :param sample: [protein_id, protein_sequence]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :return: embedding, processed_seq_len
    '''
    global global_model, global_alphabet, global_version
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
    if global_model is None or global_alphabet is None or global_version is None or global_version != version:
        if version == "3B":
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
        elif version == "650M":
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        elif version == "150M":
            global_model, global_alphabet = pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
        else:
            raise Exception("not support this version=%s" % version)
        global_version = version
    if torch.cuda.is_available() and device is not None:
        global_model = global_model.to(device)
    elif torch.cuda.is_available():
        global_model = global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    # print("llm device:", device)
    assert all(-(global_model.num_layers + 1) <= i <= global_model.num_layers for i in repr_layers)
    repr_layers = [(i + global_model.num_layers + 1) % (global_model.num_layers + 1) for i in repr_layers]
    global_model.eval()

    converter = BatchConverter(global_alphabet, truncation_seq_length)
    protein_ids, raw_seqs, tokens = converter([[protein_id, protein_seq]])
    embeddings = {}
    with torch.no_grad():
        if torch.cuda.is_available():
            tokens = tokens.to(device=device, non_blocking=True)
        try:
            out = global_model(tokens, repr_layers=repr_layers, return_contacts=False)
            truncate_len = min(truncation_seq_length, len(raw_seqs[0]))
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = out["representations"][36].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                else:
                    embedding = out["representations"][36].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out["representations"][36].to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                embedding = out["contacts"][36].to(device="cpu")[0, :, :].clone().numpy()
                embeddings["contacts"] = embedding
            if len(embeddings) > 1:
                return embeddings, protein_seq
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], protein_seq
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
    return None, None


def get_args():
    parser = argparse.ArgumentParser(description='ESM/SRM Embedding')
    # for logging
    parser.add_argument("--llm_type", type=str, default="lucaone_gplm", choices=["esm", "ESM", "lucaone_gplm"],  help="llm type")
    parser.add_argument("--llm_version", type=str, default="3B", choices=["15B", "3B", "650M", "150M"], help="llm version")
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=["matrix", "vector", "contact"], help="llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"], help="llm trunc type of seq.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094, help="truncation seq length.")
    parser.add_argument('--gpu', type=int, default=-1, help="gpu idx.")
    parser.add_argument("--fasta", type=str, default=None, help="fasta filepath")
    parser.add_argument("--seq", type=str, default=None, help="the input seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"], help="seq type")
    parser.add_argument("--save_path", type=str, default=None, help="embedding file save path")
    parser.add_argument("--matrix_add_special_token", action="store_true", help="Whether to add special token embedding in seq representation matrix")
    args = parser.parse_args()
    return args


def main(args):
    if args.gpu >= 0:
        gpu_id = args.gpu
    else:
        gpu_id = available_gpu_id()
        print("gpu_id: ", gpu_id)
    if gpu_id is None or gpu_id == -1:
        args.device = None
    else:
        args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # model.to(args.device)
    assert args.fasta is not None or args.seq is not None
    print("input seq type: %s" % args.seq_type)
    print("args device: %s" % args.device)
    embedding_type = args.embedding_type
    save_path = args.save_path
    seq_type = args.seq_type
    emb_save_path = os.path.join(save_path, args.llm_type, args.llm_version)
    print("emb save dir: %s" % emb_save_path)
    if seq_type not in ["gene", "prot"]:
        seq_type = "prot"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)

    if args.fasta:
        done = 0
        for row in fasta_reader(args.fasta):
            seq_id, seq = row[0].strip(), row[1].upper()
            if " " in seq_id or "/" in seq_id:
                emb_filename = seq_id.replace(" ", "").replace("/", "_") + ".pt"
            else:
                emb_filename = seq_id + ".pt"
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            emb, processed_seq_len = predict_embedding([seq_id, seq_type, seq],
                                                       args.trunc_type,
                                                       embedding_type,
                                                       repr_layers=[-1],
                                                       truncation_seq_length=args.truncation_seq_length,
                                                       device=args.device,
                                                       version=args.llm_version,
                                                       matrix_add_special_token=args.matrix_add_special_token
                                                       )
            # print("seq_len: %d" % len(seq))
            # print("emb shape:", embedding_info.shape)
            torch.save(emb, embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)

    elif args.seq:
        print("input seq length: %d" % len(args.seq))
        emb, processed_seq_len = predict_embedding([args.seq_id, seq_type, args.seq],
                                                   args.trunc_type,
                                                   embedding_type,
                                                   repr_layers=[-1],
                                                   truncation_seq_length=args.truncation_seq_length,
                                                   device=args.device,
                                                   version=args.llm_version,
                                                   matrix_add_special_token=args.matrix_add_special_token)
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        print(emb.shape)


if __name__ == "__main__":
    args = get_args()
    main(args)
    '''
    seq = "SDVCSSDLVNFAVEVPIEDYVAETVIAEFKETLDIQMGSNAGLRSDLMLPLLCEHRNVESNGKGRKATLARYGELDPRDQVYGPEDLNDFLGMLEDEVKKLNVVEKRL" \
          "TTRGCPQSHVLSFPTLNAINYAAFLGARHFFPDLKGQGYGDDFIGSSESREACELVLKAREGFGMMTNTTATGISRGGERGLAVFCEMVFSTLDGSLIENAKPKPVNAFFRS" \
          "LHANATMTGYVTHVEDLMDVSRMLEKRHQEVFKIIKDGVEPSKPTMAYRQTYPRFILNRIIRKDVGIDSAIATKRSAEEIFAFLESQILLFTPSSKNAPARVDPRSQSVGLLVSQ" \
          "EFHRKMSRALERYSNTKEFKFRDFPAFANAGIALLAIRKVFVDEDMEEIKGILGIRQ"
    chain = "A"
    c_alpha, c_beta = calc_distance_maps("../../pdbs/protein_1.pdb", chain, seq)
    cmap_thresh = 10
    cmap = c_alpha[chain]['contact-map']
    cmap = np.less_equal(cmap, cmap_thresh).astype(np.int32)
    print(cmap)
    print(c_alpha[chain]['contact-map'])
    print(c_beta[chain]['contact-map'])
    '''
    '''
    export CUDA_VISIBLE_DEVICES=0
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --seq DMRHRLQEKSPELADKVRFFIGDVRNLQSVRDAMHGVDYIFHAAALKQVPSCEFFPMEAVRVLGTDNVL
    
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_all.fasta \
        --save_path ../../../matrices/biolip/biolip_all
        
    export CUDA_VISIBLE_DEVICES=2
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_sita_all.fasta \
        --save_path ../../../matrices/biolip/biolip_sita_all
    
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_atp.fasta \
        --save_path ../../../matrices/biolip/biolip_atp
    
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_mg.fasta \
        --save_path ../../../matrices/biolip/biolip_mg
    
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_ca.fasta \
        --save_path ../../../matrices/biolip/biolip_ca 
        
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_hem.fasta \
        --save_path ../../../matrices/biolip/biolip_hem
    
    export CUDA_VISIBLE_DEVICES=1
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/biolip/biolip_mn.fasta \
        --save_path ../../../matrices/biolip/biolip_mn
        
    export CUDA_VISIBLE_DEVICES=3
    python predict_embedding.py \
        --llm_type esm \
        --llm_version 3B \
        --embedding_type matrix \
        --trunc_type right \
        --truncation_seq_length 4094 \
        --seq_type prot \
        --fasta ../../../data/scPDB/scPDB.fasta \
        --save_path ../../../matrices/scPDB/scPDB/
    '''




