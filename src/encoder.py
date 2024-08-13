#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 16:10
@project: LucaPCycle
@file: encoder
@desc: encoder
'''
import os
import torch
import sys
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from utils import calc_emb_filename_by_seq_id
except ImportError as e:
    from src.llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from src.utils import calc_emb_filename_by_seq_id


class Encoder(object):
    def __init__(self,
                 llm_type,
                 llm_dirpath,
                 input_type,
                 trunc_type,
                 seq_max_length,
                 prepend_bos=True,
                 append_eos=True,
                 vector_dirpath=None,
                 matrix_dirpath=None,
                 local_rank=-1,
                 **kwargs):
        self.llm_type = llm_type
        self.llm_dirpath = llm_dirpath
        self.input_type = input_type
        self.trunc_type = trunc_type
        self.seq_max_length = seq_max_length
        self.vector_dirpath = vector_dirpath
        self.matrix_dirpath = matrix_dirpath
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        if local_rank == -1 and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.cuda.is_available() and local_rank > -1:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        self.device = device
        self.seq_id_2_emb_filename = {}
        print("Encoder: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))

    def encode_single(self,
                      seq_id,
                      seq_type,
                      seq,
                      vector_filename=None,
                      matrix_filename=None,
                      label=None):
        seq_type = seq_type.strip().lower()

        # for embedding vector
        vector = None
        if self.input_type in ["vector", "seq_vector"]:
            if vector_filename is None:
                if seq is None:
                    raise Exception("seq is none and vector_filename is none")
                elif seq_type not in ["protein", "prot", "gene"]:
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    vector = self.__get_embedding__(seq_id, seq_type, seq, "vector")
            elif isinstance(vector_filename, str):
                vector = torch.load(os.path.join(self.vector_dirpath, vector_filename))
            elif isinstance(vector_filename, np.ndarray):
                vector = vector_filename
            else:
                raise Exception("vector is not filepath-str and np.ndarray")

        # for embedding matrix
        matrix = None
        if self.input_type in ["matrix", "seq_matrix"]:
            if matrix_filename is None:
                if seq is None:
                    raise Exception("seq is none and matrix_filename is none")
                elif seq_type not in ["protein", "prot", "gene"]:
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    matrix = self.__get_embedding__(seq_id, seq_type, seq, "matrix")
            elif isinstance(matrix_filename, str):
                matrix = torch.load(os.path.join(self.matrix_dirpath, matrix_filename))
            elif isinstance(matrix_filename, np.ndarray):
                matrix = matrix_filename
            else:
                raise Exception("matrix is not filepath-str and np.ndarray")

        seq = seq.upper()
        return {
            "seq_id": seq_id,
            "seq": seq,
            "seq_type": seq_type,
            "vector": vector,
            "matrix": matrix,
            "label": label
        }

    def __get_embedding__(self, seq_id, seq_type, seq, embedding_type):
        seq_type = seq_type.strip().lower()
        if "prot" not in seq_type and "gene" not in seq_type:
            raise Exception("Not support this seq_type=%s" % seq_type)
        embedding_info = None
        if seq_id in self.seq_id_2_emb_filename:
            emb_filename = self.seq_id_2_emb_filename[seq_id]
            try:
                embedding_info = torch.load(os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename))
                return embedding_info
            except Exception as e:
                print(e)
        elif embedding_type in ["bos", "vector"] and self.vector_dirpath is not None or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
            emb_filename = calc_emb_filename_by_seq_id(seq_id, embedding_type)
            embedding_path = os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename)
            if os.path.exists(embedding_path):
                try:
                    embedding_info = torch.load(embedding_path)
                    self.seq_id_2_emb_filename[seq_id] = emb_filename
                    return embedding_info
                except Exception as e:
                    print(e)
        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
        truncation_seq_length = min(len(seq), truncation_seq_length)
        while embedding_info is None:
            if self.llm_type == "esm":
                embedding_info, processed_seq = predict_embedding_esm([seq_id, seq],
                                                                      self.trunc_type,
                                                                      embedding_type,
                                                                      repr_layers=[-1],
                                                                      truncation_seq_length=truncation_seq_length,
                                                                      device=self.device)
            else:
                raise Exception("Not support the llm_type=%s" % self.llm_type)
            if embedding_info is not None:
                break
            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 - int(self.prepend_bos) - int(self.append_eos)
            truncation_seq_length = int(truncation_seq_length)
        if embedding_type in ["bos", "vector"] and self.vector_dirpath is not None or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
            emb_filename = calc_emb_filename_by_seq_id(seq_id, embedding_type)
            embedding_filepath = os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename)
            torch.save(embedding_info, embedding_filepath)
            self.seq_id_2_emb_filename[seq_id] = emb_filename
        return embedding_info

    def encode_pair(self,
                    seq_id_a,
                    seq_id_b,
                    seq_type_a,
                    seq_type_b,
                    seq_a,
                    seq_b,
                    vector_filename_a=None,
                    vector_filename_b=None,
                    matrix_filename_a=None,
                    matrix_filename_b=None,
                    label=None
                    ):
        seq_type_a = seq_type_a.strip().lower()
        seq_type_b = seq_type_b.strip().lower()

        # for embedding vector
        vector_a, vector_b = None, None
        if self.input_type in ["vector", "seq_vector"]:
            if vector_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and vector_filename_a is none")
                elif seq_type_a not in ["prot", "protein", "gene"]:
                    raise Exception("now not support embedding of the seq_type_a=%s" % seq_type_a)
                else:
                    vector_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "vector")
            elif isinstance(vector_filename_a, str):
                vector_a = torch.load(os.path.join(self.vector_dirpath, vector_filename_a))
            elif isinstance(vector_filename_a, np.ndarray):
                vector_a = vector_filename_a
            else:
                raise Exception("vector_a is not filepath-str and np.ndarray")
            if vector_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and vector_filename_b is none")
                elif seq_type_b not in ["prot", "protein", "gene"]:
                    raise Exception("now not support embedding of the seq_type_b=%s" % seq_type_b)
                else:
                    vector_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "vector")
            elif isinstance(vector_filename_b, str):
                vector_b = torch.load(os.path.join(self.vector_dirpath, vector_filename_b))
            elif isinstance(vector_filename_b, np.ndarray):
                vector_b = vector_filename_b
            else:
                raise Exception("vector_b is not filepath-str and np.ndarray")

        # for embedding matrix
        matrix_a, matrix_b = None, None
        if self.input_type in ["matrix", "seq_matrix"]:
            if matrix_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and matrix_filename_a is none")
                elif seq_type_a not in ["prot", "protein", "gene"]:
                    raise Exception("now not support embedding of the seq_type_a=%s" % seq_type_a)
                else:
                    matrix_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "matrix")
            elif isinstance(matrix_filename_a, str):
                matrix_a = torch.load(os.path.join(self.matrix_dirpath, matrix_filename_a))
            elif isinstance(matrix_filename_a, np.ndarray):
                matrix_a = matrix_filename_a
            else:
                raise Exception("matrix_a is not filepath-str and np.ndarray")
            if matrix_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and matrix_filename_b is none")
                elif seq_type_b not in ["prot", "protein", "gene"]:
                    raise Exception("now not support embedding of the seq_type_b=%s" % seq_type_b)
                else:
                    matrix_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "matrix")
            elif isinstance(matrix_filename_b, str):
                matrix_b = torch.load(os.path.join(self.matrix_dirpath, matrix_filename_b))
            elif isinstance(matrix_filename_b, np.ndarray):
                matrix_b = matrix_filename_b
            else:
                raise Exception("matrix_b is not filepath-str and np.ndarray")

        seq_a = seq_a.upper()
        seq_b = seq_b.upper()
        return {
            "seq_id_a": seq_id_a,
            "seq_a": seq_a,
            "seq_type_a": seq_type_a,
            "vector_a": vector_a,
            "matrix_a": matrix_a,
            "seq_id_b": seq_id_b,
            "seq_b": seq_b,
            "seq_type_b": seq_type_b,
            "vector_b": vector_b,
            "matrix_b": matrix_b,
            "label": label
        }




