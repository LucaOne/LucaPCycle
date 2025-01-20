#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 15:14
@project: LucaPCycle
@file: batch_converter
@desc: batch convecter
'''
import sys
import torch
from typing import Sequence

sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import gene_seq_replace
except ImportError:
    from src.utils import gene_seq_replace


class BatchConverter(object):

    def __init__(self,
                 task_level_type,
                 label_size,
                 output_mode,
                 seq_subword,
                 seq_tokenizer,
                 no_position_embeddings,
                 no_token_type_embeddings,
                 truncation_seq_length: int = None,
                 truncation_matrix_length: int = None,
                 ignore_index: int = -100,
                 padding_idx: int = 0,
                 unk_idx: int = 1,
                 cls_idx: int = 2,
                 eos_idx: int = 3,
                 mask_idx: int = 4,
                 non_ignore: bool = False,
                 seq_prepend_bos=None,
                 seq_append_eos=None,
                 matrix_prepend_bos=None,
                 matrix_append_eos=None,
                 matrix_add_special_token=None,
                 **kwargs):
        print("------BatchConverter------")
        print("BatchConverter, kwargs:")
        print(kwargs)
        self.task_level_type = task_level_type
        self.label_size = label_size
        self.output_mode = output_mode
        self.ignore_index = ignore_index
        self.non_ignore = non_ignore
        self.no_position_embeddings = no_position_embeddings
        self.no_token_type_embeddings = no_token_type_embeddings

        # for seq
        self.seq_tokenizer = seq_tokenizer
        self.seq_subword = seq_subword
        self.truncation_seq_length = truncation_seq_length
        self.truncation_matrix_length = truncation_matrix_length

        # 是否使用特殊字符
        if seq_prepend_bos is None:
            # subword 则必包含两个特殊字符
            if seq_subword is not None:
                self.seq_prepend_bos = True
            else:
                self.seq_prepend_bos = False
        else:
            self.seq_prepend_bos = seq_prepend_bos

        if seq_append_eos is None:
            if seq_subword is not None:
                self.seq_append_eos = True
            else:
                self.seq_append_eos = False
        else:
            self.seq_append_eos = seq_append_eos

        # for matrix, 是否使用特殊字符
        if matrix_prepend_bos is None:
            self.matrix_prepend_bos = False
        else:
            self.matrix_prepend_bos = matrix_prepend_bos
        if matrix_append_eos is None:
            self.matrix_append_eos = False
        else:
            self.matrix_append_eos = matrix_append_eos

        # for input matrix(输入的矩阵是否包括特殊字符）
        if matrix_add_special_token is None:
            self.matrix_add_special_token = False
        else:
            self.matrix_add_special_token = True

        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx

        print("BatchConverter Special Token Idx:")
        print("padding_idx=%d, unk_idx=%d, cls_idx=%d, eos_idx=%d, mask_idx=%d" % (
            self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx
        ))

        # for seq
        if self.seq_tokenizer is None:
            self.seq_append_len = 0
        else:
            if hasattr(self.seq_tokenizer, "prepend_bos") \
                    and self.seq_tokenizer.prepend_bos is not None:
                self.seq_prepend_bos = self.seq_tokenizer.prepend_bos
            if hasattr(self.seq_tokenizer, "append_eos") \
                    and self.seq_tokenizer.append_eos is not None:
                self.seq_append_eos = self.seq_tokenizer.append_eos

            if hasattr(self.seq_tokenizer, "padding_idx") \
                    and self.seq_tokenizer.padding_idx is not None:
                self.padding_idx = self.seq_tokenizer.padding_idx
            if hasattr(self.seq_tokenizer, "unk_idx") \
                    and self.seq_tokenizer.unk_idx is not None:
                self.unk_idx = self.seq_tokenizer.unk_idx
            if hasattr(self.seq_tokenizer, "cls_idx") \
                    and self.seq_tokenizer.cls_idx is not None:
                self.cls_idx = self.seq_tokenizer.cls_idx
            if hasattr(self.seq_tokenizer, "eos_idx") \
                    and self.seq_tokenizer.eos_idx is not None:
                self.eos_idx = self.seq_tokenizer.eos_idx
            if hasattr(self.seq_tokenizer, "mask_idx") \
                    and self.seq_tokenizer.mask_idx is not None:
                self.mask_idx = self.seq_tokenizer.mask_idx

            if hasattr(self.seq_tokenizer, "all_special_token_idx_list"):
                self.all_special_token_idx_list = self.seq_tokenizer.all_special_token_idx_list
            else:
                self.all_special_token_idx_list = [self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx]
            self.seq_append_len = int(self.seq_prepend_bos) + int(self.seq_append_eos)
        print("BatchConverter Special Token Idx:")
        print("padding_idx=%d, unk_idx=%d, cls_idx=%d, eos_idx=%d, mask_idx=%d" % (
            self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx
        ))
        if self.truncation_seq_length:
            # 减去特殊字符之后的长度
            self.truncation_seq_length -= self.seq_append_len
        else:
            self.truncation_seq_length = 0
        # for matrix
        self.matrix_append_len = int(self.matrix_prepend_bos) + int(self.matrix_append_eos)
        if self.truncation_matrix_length:
            self.truncation_matrix_length -= self.matrix_append_len
        else:
            self.truncation_matrix_length = 0

        if "batch_with_seq_ids" in kwargs and kwargs["batch_with_seq_ids"]:
            self.batch_with_seq_ids = kwargs["batch_with_seq_ids"]
        else:
            self.batch_with_seq_ids = False

        print("BatchConverter: truncation_seq_length=%d, truncation_matrix_length=%d" % (self.truncation_seq_length, self.truncation_matrix_length))
        print("BatchConverter: seq_prepend_bos=%r, seq_append_eos=%r" % (self.seq_prepend_bos, self.seq_append_eos))
        print("BatchConverter: matrix_prepend_bos=%r, matrix_append_eos=%r" % (self.matrix_prepend_bos, self.matrix_append_eos))
        print("BatchConverter: matrix_add_special_token=%r" % self.matrix_add_special_token)

        self.input_type = None
        if "input_type" in kwargs and kwargs["input_type"]:
            self.input_type = kwargs["input_type"]

        self.trunc_type = "right"
        if "trunc_type" in kwargs and kwargs["trunc_type"]:
            self.trunc_type = kwargs["trunc_type"]
            print("BatchConverter: trunc_type=%s" % self.trunc_type)
        print("-" * 50)

    def __parse_label__(self, max_length, task_level_type, label_size, output_mode, label):
        if isinstance(label, str):
            label = eval(label)
        # 需要是padding长度
        cur_len = max_length
        if task_level_type in ["token_level", "structure_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.prepend_bos:
                    begin_idx = 1
                if self.append_eos:
                    end_idx = cur_len - 1
                for idx, item in enumerate(label):
                    idx += begin_idx
                    if idx >= end_idx:
                        break
                    if output_mode in ["multi_label", "multi-label"]:
                        for v in item:
                            new_label[idx][v] = 1
                    else:
                        new_label[idx] = item
        elif task_level_type == "span_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.prepend_bos:
                    begin_idx = 1
                if self.append_eos:
                    end_idx = cur_len - 1
                for item in label:
                    for idx in range(item[0], item[1] + 1, 1):
                        idx += begin_idx
                        if idx >= end_idx:
                            break
                        if output_mode in ["multi_label", "multi-label"]:
                            new_label[idx][item[2]] = 1
                        else:
                            new_label[idx] = item[2]
        elif task_level_type in ["seq_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * label_size
                new_label = []
                for _ in range(label_size):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            else:
                # N * 1
                new_label = [0 if self.non_ignore else self.ignore_index]
            if output_mode in ["multi_label", "multi-label"]:
                if label is not None and len(label) > 0:
                    for v in label:
                        new_label[int(v)] = 1
            else:
                if label is not None and len(str(label)) > 0:
                    if isinstance(label, str):
                        new_label = [int(label)]
                    elif isinstance(label, list):
                        new_label = [int(label[0])]
                    else:
                        new_label = [label]
        else:
            raise Exception("Not support task_level_type=%s" % task_level_type)

        return new_label

    def __seq_encode__(self, batch_size, seqs):
        """
        该函数不加特殊字符[CLS]与[SEP]
        :param batch_size:
        :param seqs:
        :return:
        """
        if self.seq_subword:
            seq_encoded_list = []
            for seq_str in seqs:
                seq_to_list = self.seq_subword.process_line(seq_str.upper()).split(" ")
                seq = " ".join(seq_to_list)
                inputs = self.seq_tokenizer.encode_plus(
                    seq,
                    None,
                    add_special_tokens=False,
                    max_length=self.truncation_seq_length,
                    truncation=True
                )
                seq_encoded_list.append(inputs["input_ids"])
        else:
            seq_encoded_list = [self.seq_tokenizer.encode(seq_str.upper()) for seq_str in seqs]
            # 该长度已经减去了需要增加的特殊字符的个数
            if self.truncation_seq_length:
                seq_encoded_list = [encoded[:self.truncation_seq_length] for encoded in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.seq_prepend_bos) + int(self.seq_append_eos)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.padding_idx)

        token_type_ids = None
        if not self.no_position_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.padding_idx)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)

        return seq_encoded_list, input_ids, position_ids, token_type_ids, attention_masks, max_len

    def __vector_encode__(self, batch_size, vectors):
        """
        vector encoder
        :param batch_size:
        :param vectors:
        :return:
        """
        embedding_vector_dim = vectors[0].shape[0]
        filled_vectors = torch.empty(
            (
                batch_size,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_vectors.fill_(0.0)
        return filled_vectors, 1

    def __matrix_encode__(self, batch_size, matrices):
        """
        该函数不加特殊字符[CLS]与[SEP]的向量
        :param batch_size:
        :param matrices:
        :return:
        """
        max_len = max(matrix.shape[0] for matrix in matrices)
        # 表征有特殊字符，并且 在模型中需要使用，则实际长度-2
        if self.matrix_add_special_token and self.matrix_prepend_bos and self.matrix_append_eos:
            max_len -= 2
        if self.truncation_matrix_length:
            max_len = min(max_len, self.truncation_matrix_length)
        max_len = max_len + int(self.matrix_prepend_bos) + int(self.matrix_append_eos)
        embedding_vector_dim = matrices[0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                max_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, max_len

    def __call_single__(self, batch_size, seq_types, seqs, vectors, matrices, labels):
        max_length = sys.maxsize
        input_ids, position_ids, token_type_ids, seq_attention_masks = None, None, None, None
        seq_part_of_input = False
        if seqs:
            new_seqs = []
            for seq_idx, seq_type in enumerate(seq_types):
                if seq_type == "prot":
                    new_seqs.append(seqs[seq_idx].upper())
                else:
                    raise Exception("not support the seq_type=%s" % seq_type)

            # seq_encoded_list没有加特殊字符，input_ids标志位来占位， seq_max_length 根据标志位来加特殊字符长度
            seq_encoded_list, input_ids, position_ids, token_type_ids, seq_attention_masks, seq_max_length = self.__seq_encode__(
                batch_size=batch_size,
                seqs=new_seqs
            )
            max_length = min(max_length, seq_max_length)
            seq_part_of_input = True

        encoded_vectors = None
        vector_part_of_input = False
        if vectors is not None and len(vectors) > 0:
            encoded_vectors, vector_max_length = self.__vector_encode__(
                batch_size=batch_size,
                vectors=vectors
            )
            vector_part_of_input = True

        encoded_matrices, matrix_attention_masks = None, None
        matrix_part_of_input = False
        if matrices is not None and len(matrices) > 0:

            # 根据标记位填充，根据标记位填充，句子数量，根据标记位是否加上特殊字符长度
            encoded_matrices, matrix_attention_masks, matrix_max_length = self.__matrix_encode__(
                batch_size=batch_size,
                matrices=matrices
            )
            max_length = min(max_length, matrix_max_length)
            matrix_part_of_input = True

        has_label = False
        if labels:
            has_label = True

        new_labels = []
        num_sentences = 1
        sentence_length = 1
        for sample_idx in range(batch_size):
            # seq
            if seq_part_of_input:
                if self.seq_prepend_bos:
                    input_ids[sample_idx, 0] = self.cls_idx

                seq_encoded = seq_encoded_list[sample_idx]
                real_seq_len = len(seq_encoded)

                seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
                input_ids[sample_idx, int(self.seq_prepend_bos): real_seq_len + int(self.seq_prepend_bos)] = seq_tensor
                cur_sentence_length = int(self.seq_prepend_bos) + real_seq_len + int(self.seq_prepend_bos)
                if cur_sentence_length > sentence_length:
                    sentence_length = cur_sentence_length

                if self.seq_append_eos:
                    input_ids[sample_idx, real_seq_len + int(self.seq_prepend_bos)] = self.eos_idx

                cur_len = int(self.seq_prepend_bos) + real_seq_len + int(self.seq_append_eos)

                if not self.no_position_embeddings:
                    for pos_idx in range(0, cur_len):
                        position_ids[sample_idx, pos_idx] = pos_idx

                if not self.no_token_type_embeddings:
                    seq_type = seq_types[sample_idx]
                    # assert seq_type in ["gene", "prot", "molecule"]
                    if "gene" in seq_type:
                        type_value = 0
                    elif "prot" in seq_type:
                        type_value = 1
                    else:
                        raise Exception("Not support the seq_type=%s" % seq_type)

                    for pos_idx in range(0, cur_len):
                        token_type_ids[sample_idx, pos_idx] = type_value

                seq_attention_masks[sample_idx, 0: cur_len] = 1

            # vector
            if vector_part_of_input:
                encoded_vectors[sample_idx, :] = torch.tensor(vectors[sample_idx], dtype=torch.float32)

            # matrix
            if matrix_part_of_input:
                matrix_encoded = matrices[sample_idx]
                # 不包括特殊字符长度（如果有则去掉）
                if self.matrix_add_special_token and self.matrix_prepend_bos and self.matrix_append_eos:
                    real_matrix_len = matrix_encoded.shape[0] - 2
                else:
                    real_matrix_len = matrix_encoded.shape[0]
                # 是否截断
                real_matrix_len = min(real_matrix_len, self.truncation_matrix_length)
                # print("real_matrix_len: %d" % real_matrix_len)
                matrix = torch.tensor(matrix_encoded, dtype=torch.float32)

                if self.matrix_add_special_token and self.matrix_prepend_bos and self.matrix_append_eos:
                    # embedding矩阵中有特殊字符，并且模型中需要使用
                    encoded_matrices[sample_idx, 1: real_matrix_len + 1] = matrix[1: real_matrix_len + 1]
                    encoded_matrices[sample_idx, 0] = matrix[0]
                    encoded_matrices[sample_idx, real_matrix_len + 1] = matrix[-1]
                    matrix_attention_masks[sample_idx, 0: real_matrix_len + 2] = 1
                    cur_sentence_length = real_matrix_len + 2
                elif self.matrix_add_special_token:
                    # embedding矩阵中有特殊字符，但模型中不需要使用（已经进行了裁剪）
                    encoded_matrices[sample_idx, 0: real_matrix_len] = matrix[0: real_matrix_len]
                    matrix_attention_masks[sample_idx, 0: real_matrix_len] = 1
                    cur_sentence_length = real_matrix_len
                elif self.matrix_prepend_bos and self.matrix_append_eos:
                    # 首尾使用zero的
                    encoded_matrices[sample_idx, 1: real_matrix_len + 1] = matrix[0: real_matrix_len]
                    # 并且mask为0
                    matrix_attention_masks[sample_idx, 1: real_matrix_len + 1] = 1
                    cur_sentence_length = real_matrix_len + 2
                else:
                    encoded_matrices[sample_idx, 0: real_matrix_len] = matrix[0: real_matrix_len]
                    matrix_attention_masks[sample_idx, 0: real_matrix_len] = 1
                    cur_sentence_length = real_matrix_len
                if cur_sentence_length > sentence_length:
                    sentence_length = cur_sentence_length

            if has_label:
                new_labels.append(
                    self.__parse_label__(max_length=max_length,
                                         task_level_type=self.task_level_type,
                                         label_size=self.label_size,
                                         output_mode=self.output_mode,
                                         label=labels[sample_idx]))

        if new_labels is not None and new_labels:
            if self.output_mode in ["regression"]:
                labels = torch.tensor(new_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(new_labels, dtype=torch.int64)
        else:
            labels = None

        return input_ids, \
               position_ids, \
               token_type_ids, \
               seq_attention_masks, \
               encoded_vectors, \
               encoded_matrices, \
               matrix_attention_masks, \
               num_sentences, \
               sentence_length, \
               labels

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        # pair
        if "seq_id_a" in raw_batch[0] and "seq_id_b" in raw_batch[0]:
            res = {}
            seq_ids_a = []
            seq_types_a = []
            seqs_a = []
            vectors_a = []
            matrices_a = []

            seq_ids_b = []
            seq_types_b = []
            seqs_b = []
            vectors_b = []
            matrices_b = []

            labels = []
            for item in raw_batch:
                seq_ids_a.append(item["seq_id_a"])
                seq_types_a.append(item["seq_type_a"])
                if item["seq_a"] is not None:
                    seqs_a.append(item["seq_a"])
                if item["vector_a"] is not None:
                    vectors_a.append(item["vector_a"])
                if item["matrix_a"] is not None:
                    matrices_a.append(item["matrix_a"])

                seq_ids_b.append(item["seq_id_b"])
                seq_types_b.append(item["seq_type_b"])
                if item["seq_b"] is not None:
                    seqs_b.append(item["seq_b"])
                if item["vector_b"] is not None:
                    vectors_b.append(item["vector_b"])
                if item["matrix_b"] is not None:
                    matrices_b.append(item["matrix_b"])
                if "label" in item and item["label"] is not None:
                    labels.append(item["label"])
            # embedding 矩阵有特殊字符，如果不使用则去掉首尾的特殊字符
            new_matrices_a = []
            if matrices_a:
                for seq_idx_a, seq_type_a in enumerate(seq_types_a):
                    if "molecule" in seq_type_a:
                        if self.atom_matrix_add_special_token \
                                and (not self.atom_matrix_prepend_bos or not self.atom_matrix_append_eos):
                            new_matrices_a.append(matrices_a[seq_idx_a][1:-1])
                    else:
                        if self.matrix_add_special_token \
                                and (not self.matrix_prepend_bos or not self.matrix_append_eos):
                            new_matrices_a.append(matrices_a[seq_idx_a][1:-1])
                if new_matrices_a and len(new_matrices_a) > 0:
                    matrices_a = new_matrices_a
            input_ids_a, position_ids_a, token_type_ids_a, seq_attention_masks_a, encoded_vectors_a, encoded_matrices_a, matrix_attention_masks_a, num_sentences_a, sentence_length_a, labels \
                = self.__call_single__(batch_size, seq_types_a, seqs_a, vectors_a, matrices_a, labels)
            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids_a": input_ids_a,
                    "position_ids_a": position_ids_a,
                    "token_type_ids_a": token_type_ids_a,
                    "seq_attention_masks_a": seq_attention_masks_a,
                    "vectors_a": encoded_vectors_a,
                    "matrices_a": encoded_matrices_a,
                    "matrix_attention_masks_a": matrix_attention_masks_a,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids_a": seq_ids_a
                    })
            else:
                res.update({
                    "input_ids_a": input_ids_a,
                    "position_ids_a": position_ids_a,
                    "token_type_ids_a": token_type_ids_a,
                    "seq_attention_masks_a": seq_attention_masks_a,
                    "vectors_a": encoded_vectors_a,
                    "matrices_a": encoded_matrices_a,
                    "matrix_attention_masks_a": matrix_attention_masks_a,
                    "num_sentences_a": num_sentences_a,
                    "sentence_length_a": sentence_length_a,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids_a": seq_ids_a
                    })
            # embedding 矩阵有特殊字符，如果不使用则去掉首尾的特殊字符
            new_matrices_b = []
            if matrices_b:
                for seq_idx_b, seq_type_b in enumerate(seq_types_b):
                    if "molecule" in seq_type_b:
                        if self.atom_matrix_add_special_token \
                                and (not self.atom_matrix_prepend_bos or not self.atom_matrix_append_eos):
                            new_matrices_b.append(matrices_b[seq_idx_b][1:-1])
                    else:
                        if self.matrix_add_special_token \
                                and (not self.matrix_prepend_bos or not self.matrix_append_eos):
                            new_matrices_b.append(matrices_b[seq_idx_b][1:-1])
                if new_matrices_b and len(new_matrices_b) > 0:
                    matrices_b = new_matrices_b
            input_ids_b, position_ids_b, token_type_ids_b, seq_attention_masks_b, encoded_vectors_b, encoded_matrices_b, matrix_attention_masks_b, num_sentences_b, sentence_length_b,  _ \
                = self.__call_single__(batch_size, seq_types_b, seqs_b, vectors_b, matrices_b, labels=None)
            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids_b": input_ids_b,
                    "position_ids_b": position_ids_b,
                    "token_type_ids_b": token_type_ids_b,
                    "seq_attention_masks_b": seq_attention_masks_b,
                    "vectors_b": encoded_vectors_b,
                    "matrices_b": encoded_matrices_b,
                    "matrix_attention_masks_b": matrix_attention_masks_b
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids_b": seq_ids_b
                    })
            else:
                res.update({
                    "input_ids_b": input_ids_b,
                    "position_ids_b": position_ids_b,
                    "token_type_ids_b": token_type_ids_b,
                    "seq_attention_masks_b": seq_attention_masks_b,
                    "vectors_b": encoded_vectors_b,
                    "matrices_b": encoded_matrices_b,
                    "num_sentences_b": num_sentences_b,
                    "sentence_length_b": sentence_length_b,
                    "matrix_attention_masks_b": matrix_attention_masks_b
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids_b": seq_ids_b
                    })
            return res
        else:
            res = {}
            seq_ids = []
            seq_types = []
            seqs = []
            vectors = []
            matrices = []
            labels = []
            for item in raw_batch:
                seq_ids.append(item["seq_id"])
                seq_types.append(item["seq_type"])
                if item["seq"] is not None:
                    seqs.append(item["seq"])
                if item["vector"] is not None:
                    vectors.append(item["vector"])
                if item["matrix"] is not None:
                    matrices.append(item["matrix"])
                if item["label"] is not None:
                    labels.append(item["label"])
            # embedding 矩阵有特殊字符，如果不使用则去掉首尾的特殊字符
            new_matrices = []
            if matrices:
                for seq_idx, seq_type in enumerate(seq_types):
                    if "molecule" in seq_type:
                        if self.atom_matrix_add_special_token \
                                and (not self.atom_matrix_prepend_bos or not self.atom_matrix_append_eos):
                            new_matrices.append(matrices[seq_idx][1:-1])
                    else:
                        if self.matrix_add_special_token \
                                and (not self.matrix_prepend_bos or not self.matrix_append_eos):
                            new_matrices.append(matrices[seq_idx][1:-1])
                if new_matrices and len(new_matrices) > 0:
                    matrices = new_matrices
            input_ids, position_ids, token_type_ids, seq_attention_masks, encoded_vectors, encoded_matrices, matrix_attention_masks, num_sentences, sentence_length, labels = self.__call_single__(
                batch_size, seq_types, seqs, vectors, matrices, labels=labels)

            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "token_type_ids": token_type_ids,
                    "seq_attention_masks": seq_attention_masks,
                    "vectors": encoded_vectors,
                    "matrices": encoded_matrices,
                    "matrix_attention_masks": matrix_attention_masks,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids": seq_ids
                    })
            else:
                res.update({
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "token_type_ids": token_type_ids,
                    "seq_attention_masks": seq_attention_masks,
                    "vectors": encoded_vectors,
                    "matrices": encoded_matrices,
                    "matrix_attention_masks": matrix_attention_masks,
                    "num_sentences": num_sentences,
                    "sentence_length": sentence_length,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
                if self.batch_with_seq_ids:
                    res.update({
                        "seq_ids": seq_ids
                    })
            return res
