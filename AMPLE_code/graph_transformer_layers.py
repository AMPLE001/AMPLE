import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f

from dgl.nn import GatedGraphConv, GraphConv, AvgPooling, MaxPooling,  RelGraphConv
import dgl
import dgl.function as fn
import math
import numpy as np
"""
    GraphNorm

"""
class Norm(nn.Module):

    def __init__(self, norm_type = 'gn', hidden_dim=100, print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        tensor = tensor
        batch_list = graph.batch_num_nodes().float()
        batch_size = len(batch_list)
        batch_list = torch.tensor(batch_list, device = torch.device('cuda:0'))
        batch_list = batch_list.long()
        
        batch_index = torch.arange(batch_size, device = torch.device('cuda:0')).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:], device = torch.device('cuda:0'))
        
        
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        
        result = self.weight * sub / std + self.bias
        return result.to(torch.device('cuda:0'))

"""
    Util functions
"""

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}
        #return {field: torch.exp((edges.data[field] / scale_constant))}
    return func


"""
    MultiHeadAttentionLayer Layer
    多头注意力机制

"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias = True):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        num_steps = 6
        max_edge_types = 5
        self.feature_Q = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, low_mem = False, dropout = 0.1)
        self.feature_K = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, low_mem = False, dropout = 0.1)
        self.feature_V = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, low_mem = False, dropout = 0.1)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):
        #g_Q = g
        #g_V = g
        #print(h.shape)
        feature_Q = self.feature_Q(g, h, e)
        #print(feature_Q.shape)
        feature_K = self.feature_K(g, h, e)
        feature_V = self.feature_V(g, h, e)
        #print(feature_Q.shape)
        Q_h = feature_Q#self.Q(feature_Q)
        K_h = feature_K#self.K(feature_K)
        V_h = feature_V#self.V(feature_V)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        #head_out = g.ndata['wV'] / g.ndata['z']
        head_out= g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        return head_out


"""
    Graph Transformer Layer

"""
class GraphTransformerLayer(nn.Module):
    """
        Param:
    """


    def __init__(self, input_dim, output_dim, max_edge_types, num_heads, num_steps=8, dropout=0.0, layer_norm=False, batch_norm=True, residual=False,
                 use_bias=True):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps



        
        self.attention = MultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)

        self.O = nn.Linear(output_dim, output_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(output_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)
            self.Graph_norm1 = Norm(hidden_dim=output_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(output_dim, output_dim * 2)
        self.FFN_layer2 = nn.Linear(output_dim * 2, output_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(output_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.Graph_norm2 = Norm(hidden_dim=output_dim)

    def forward(self, graph, h, e):
        
        h_in1 = h  # for first residual connection
        #_in1= h_in1.view(-1, self.out_channels)
        # multi-head attention out
        if self.batch_norm:
            #h = self.batch_norm1(h)
            h = self.Graph_norm1(graph, h)
        #多头注意力机制网络 输入 graph 和 h ,仅使用单头，因为每个输入的维度不一样

        attn_out = self.attention(graph, h, e)

        h = attn_out.view(-1, self.out_channels)


        h = f.dropout(h, self.dropout, training=self.training)

        #dense层 nn.Linear(output_dim, output_dim)
        h = self.O(h)

        #是否使用残差连接 目前是矩阵模块未对齐
        if self.residual:


            h = h_in1 + h  # residual connection


        if self.layer_norm:
            h = self.layer_norm1(h)
   



        h_in2 = h  # for second residual connection
        if self.batch_norm:
            #h = self.batch_norm2(h)
            h = self.Graph_norm2(graph, h)
        # FFN
        # 连续的两个前向网络 两个前向网络后 输出的维度和输入相同
        h = self.FFN_layer1(h)
        h = f.relu(h)
        h = f.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)


            




        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
