#!/usr/bin/env python
# encoding: utf-8
'''
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/19 14:37
@project: LucaPCycle
@file: gcn
@desc: GCN layer of our model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from layers import GraphAttentionLayer, SpGraphAttentionLayer
except ImportError:
    from src.lucaprot.models.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, dropout, alpha, nheads):
        '''
        :param feature_size:
        :param hidden_size:
        :param output_size:
        :param dropout:
        :param alpha:
        :param nheads:
        '''
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionLayer(feature_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x: (batch_size, N, feature_size), adj:(batch_size, N, N)
        x = F.dropout(x, self.dropout, training=self.training)
        # (batch_size, N, hidden_size)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # (batch_size, N, 4 * hidden_size)
        x = F.dropout(x, self.dropout, training=self.training)
        # (batch_size, N, 4 * hidden_size)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x


class SpGAT(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(feature_size,
                                                 hidden_size,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)])

        self.out_att = SpGraphAttentionLayer(hidden_size * nheads,
                                             output_size,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=-1)
        return x