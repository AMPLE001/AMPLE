import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import dgl
import dgl.function as fn
import math
import numpy as np
from graph_transformer_layers import GraphTransformerLayer
from mlp_readout import MLPReadout

from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, RelGraphConv

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, small_kernel, large_kernel, stride, groups):
        super().__init__()
        self.large_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size= large_kernel, stride= stride, padding= large_kernel // 2, groups=groups, dilation=1, bias = True)
        self.large_bn = torch.nn.BatchNorm1d(out_channels)
        self.small_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size= small_kernel, stride= stride, padding= small_kernel // 2, groups=groups, dilation=1)
        self.small_bn = torch.nn.BatchNorm1d(out_channels)


    def forward(self, inputs):
        large_out = self.large_conv(inputs)
        large_out = self.large_bn(large_out)
        small_out = self.small_conv(inputs)
        small_out = self.small_bn(small_out)
        return large_out + small_out

class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.gcn = GraphConv(in_feats=input_dim, out_feats=output_dim)
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        self.gtn =  nn.ModuleList([GraphTransformerLayer(input_dim, output_dim, num_heads = num_head,
                                              dropout = 0.2,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
                                   for _ in range (n_layers - 1)])
        self.MPL_layer = MLPReadout(output_dim, 2)
        self.sigmoid = nn.Sigmoid()
        
        
        ffn_ratio = 2
        self.concat_dim = output_dim
        small_kernel = 3
        large_kernel = 11
        self.RepLK = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
            torch.nn.ReLU(),
            ReparamLargeKernelConv(in_channels = self.concat_dim * ffn_ratio, out_channels = self.concat_dim * ffn_ratio, small_kernel = small_kernel, large_kernel = large_kernel, stride = 1, groups = self.concat_dim * ffn_ratio),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
        )
        k = 3

        self.Avgpool1 =  torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride= k),
            torch.nn.Dropout(0.1)
        )
        self.ConvFFN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.concat_dim),
            torch.nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(self.concat_dim  * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1),
        )
        self.Avgpool2 =  torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride= k),
            torch.nn.Dropout(0.1)
        )

        

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device('cuda:0'))
        for conv in self.gtn:
            features = conv(graph, features, edge_types)
        outputs = batch.de_batchify_graphs(features)
        
        outputs = outputs.transpose(1, 2)
        ''' 
              Layer1
        '''
        outputs += self.RepLK(outputs)
        outputs = self.Avgpool1(outputs)
        outputs += self.ConvFFN(outputs)
        outputs = self.Avgpool2(outputs)
        '''
              Layer2
        ''' 
        outputs = outputs.transpose(1, 2)
        outputs = self.MPL_layer(outputs.sum(dim=1))
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs


