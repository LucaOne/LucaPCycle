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
                 prepend_bos=None,
                 append_eos=None,
                 **kwargs):
        print("BatchConverter, kwargs:")
        print(kwargs)
        self.task_level_type = task_level_type
        self.label_size = label_size
        self.output_mode = output_mode
        self.seq_tokenizer = seq_tokenizer
        self.seq_subword = seq_subword
        self.ignore_index = ignore_index
        self.non_ignore = non_ignore
        self.truncation_seq_length = truncation_seq_length
        self.truncation_matrix_length = truncation_matrix_length

        if prepend_bos is None:
            if seq_subword is not None:
                self.prepend_bos = True
            else:
                self.prepend_bos = False
        else:
            self.prepend_bos = prepend_bos
        if append_eos is None:
            if seq_subword is not None:
                self.append_eos = True
            else:
                self.append_eos = False
        else:
            self.append_eos = append_eos

        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx
        if self.seq_tokenizer is None:
            self.append_len = 0
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
            self.append_len = int(self.prepend_bos) + int(self.append_eos)
        # 减去特殊字符之后的长度
        self.truncation_seq_length -= self.append_len
        self.truncation_matrix_length -= self.append_len

        self.no_position_embeddings = no_position_embeddings
        self.no_token_type_embeddings = no_token_type_embeddings
        print("BatchConverter: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))


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
        '''
        该函数不加特殊字符[CLS]与[SEP]
        :param batch_size:
        :param seqs:
        :return:
        '''
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
        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)
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
        '''
        该函数不加特殊字符[CLS]与[SEP]的向量
        :param batch_size:
        :param matrices:
        :return:
        '''
        max_len = max(matrix.shape[0] for matrix in matrices)
        if self.truncation_matrix_length:
            max_len = min(max_len, self.truncation_matrix_length)
        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)
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
                if seq_type == "gene":
                    new_seqs.append(gene_seq_replace(seqs[seq_idx].upper()))
                else:
                    new_seqs.append(seqs[seq_idx].upper())
            seq_encoded_list, input_ids, position_ids, token_type_ids, seq_attention_masks, seq_max_length = self.__seq_encode__(
                batch_size=batch_size, seqs=new_seqs)
            max_length = min(max_length, seq_max_length)
            seq_part_of_input = True
        else:
            seq_encoded_list = None

        encoded_vectors = None
        vector_part_of_input = False
        if vectors:

            encoded_vectors, vector_max_length = self.__vector_encode__(batch_size=batch_size, vectors=vectors)
            # max_length = min(max_length, vector_max_length)
            vector_part_of_input = True

        encoded_matrices, matrix_attention_masks = None, None
        matrix_part_of_input = False
        if matrices:
            encoded_matrices, matrix_attention_masks, matrix_max_length = self.__matrix_encode__(batch_size=batch_size,
                                                                                                     matrices=matrices)
            max_length = min(max_length, matrix_max_length)
            matrix_part_of_input = True
        has_label = False
        if labels:
            has_label = True

        new_labels = []
        for sample_idx in range(batch_size):
            # seq
            if seq_part_of_input:
                if self.prepend_bos:
                    input_ids[sample_idx, 0] = self.cls_idx
                seq_encoded = seq_encoded_list[sample_idx]
                real_seq_len = len(seq_encoded)

                seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
                input_ids[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = seq_tensor

                if self.append_eos:
                    input_ids[sample_idx, real_seq_len + int(self.prepend_bos)] = self.eos_idx

                cur_len = int(self.prepend_bos) + real_seq_len + int(self.append_eos)

                if not self.no_position_embeddings:
                    for pos_idx in range(0, cur_len):
                        position_ids[sample_idx, pos_idx] = pos_idx

                if not self.no_token_type_embeddings:
                    seq_type = seq_types[sample_idx]
                    if seq_type == "gene":
                        type_value = 0
                    else:
                        type_value = 1

                    for pos_idx in range(0, cur_len):
                        token_type_ids[sample_idx, pos_idx] = type_value

                seq_attention_masks[sample_idx, 0: cur_len] = 1

            # vector
            if vector_part_of_input:
                encoded_vectors[sample_idx, :] = torch.tensor(vectors[sample_idx], dtype=torch.float32)

            # matrix
            if matrix_part_of_input:
                matrix_encoded = matrices[sample_idx]
                real_seq_len = matrix_encoded.shape[0]

                real_seq_len = min(real_seq_len, self.truncation_matrix_length)
                # print("real_seq_len: %d" % real_seq_len)

                matrix = torch.tensor(matrix_encoded, dtype=torch.float32)
                encoded_matrices[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = matrix[0: real_seq_len]
                matrix_attention_masks[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = 1

            if has_label:
                new_labels.append(
                    self.__parse_label__(max_length, self.task_level_type,
                                         self.label_size, self.output_mode, labels[sample_idx]))
        if new_labels is not None and new_labels:
            if self.output_mode in ["regression"]:
                labels = torch.tensor(new_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(new_labels, dtype=torch.int64)
        else:
            labels = None

        return input_ids, position_ids, token_type_ids, seq_attention_masks, encoded_vectors, encoded_matrices, matrix_attention_masks, labels

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        # pair
        if "seq_id_a" in raw_batch[0] and "seq_id_b" in raw_batch[0]:
            res = {}
            # seq_ids_a = []
            seq_types_a = []
            seqs_a = []
            vectors_a = []
            matrices_a = []

            # seq_ids_b = []
            seq_types_b = []
            seqs_b = []
            vectors_b = []
            matrices_b = []

            labels = []
            for item in raw_batch:
                # seq_ids_a.append(item["seq_id_a"])
                seq_types_a.append(item["seq_type_a"])
                if item["seq_a"] is not None:
                    seqs_a.append(item["seq_a"])
                if item["vector_a"] is not None:
                    vectors_a.append(item["vector_a"])
                if item["matrix_a"] is not None:
                    matrices_a.append(item["matrix_a"])

                # seq_ids_b.append(item["seq_id_b"])
                seq_types_b.append(item["seq_type_b"])
                if item["seq_b"] is not None:
                    seqs_b.append(item["seq_b"])
                if item["vector_b"] is not None:
                    vectors_b.append(item["vector_b"])
                if item["matrix_b"] is not None:
                    matrices_b.append(item["matrix_b"])
                if "label" in item and item["label"] is not None:
                    labels.append(item["label"])
            input_ids_a, position_ids_a, token_type_ids_a, seq_attention_masks_a, encoded_vectors_a, encoded_matrices_a, matrix_attention_masks_a, labels \
                = self.__call_single__(batch_size, seq_types_a, seqs_a, vectors_a, matrices_a, labels)

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
            input_ids_b, position_ids_b, token_type_ids_b, seq_attention_masks_b, encoded_vectors_b, encoded_matrices_b, matrix_attention_masks_b,  _ \
                = self.__call_single__(batch_size, seq_types_b, seqs_b, vectors_b, matrices_b, labels=None)
            res.update({
                "input_ids_b": input_ids_b,
                "position_ids_b": position_ids_b,
                "token_type_ids_b": token_type_ids_b,
                "seq_attention_masks_b": seq_attention_masks_b,
                "vectors_b": encoded_vectors_b,
                "matrices_b": encoded_matrices_b,
                "matrix_attention_masks_b": matrix_attention_masks_b
            })
            return res
        else:
            res = {}
            # seq_ids = []
            seq_types = []
            seqs = []
            vectors = []
            matrices = []
            labels = []
            for item in raw_batch:
                # seq_ids.append(item["seq_id"])
                seq_types.append(item["seq_type"])
                if item["seq"] is not None:
                    seqs.append(item["seq"])
                if item["vector"] is not None:
                    vectors.append(item["vector"])
                if item["matrix"] is not None:
                    matrices.append(item["matrix"])
                if item["label"] is not None:
                    labels.append(item["label"])
            input_ids, position_ids, token_type_ids, seq_attention_masks, encoded_vectors, encoded_matrices, matrix_attention_masks, labels = self.__call_single__(
                batch_size, seq_types, seqs, vectors, matrices, labels=labels)

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
            return res
