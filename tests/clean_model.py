
import torch
from torch import nn, Tensor
import torch.optim as optim
# from egnn_pytorch import EGNN
import egcnModel
from gcnLayer import GraphConvolution, GlobalPooling
#from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
#from se3_transformer_pytorch.irr_repr import rot
#from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
import wandb
import json
#from sklearn.neighbors import NearestNeighbors
#from datasets import TUMDataset

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from torch_geometric.nn import global_max_pool, MessagePassing
# from torch_geometric.data import Datao8
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import SamplePoints, KNNGraph
import torch_geometric.transforms as T
from torch_cluster import knn_graph
# from utils.SE3 import *
import pickle
# import open3d as o3d

torch.cuda.manual_seed(2)

# class TransformationLoss(nn.Module):
#     def __init__(self, re_thre=15, te_thre=30):
#         super(TransformationLoss, self).__init__()
#         self.re_thre = re_thre  # rotation error threshold (deg)
#         self.te_thre = te_thre  # translation error threshold (cm)

#     def forward(self, trans, gt_trans, src_keypts, tgt_keypts, probs):
#         """
#         Transformation Loss
#         Inputs:
#             - trans:      [bs, 4, 4] SE3 transformation matrices
#             - gt_trans:   [bs, 4, 4] ground truth SE3 transformation matrices
#             - src_keypts: [bs, num_corr, 3]
#             - tgt_keypts: [bs, num_corr, 3]
#             - probs:     [bs, num_corr] predicted inlier probability
#         Outputs:
#             - loss     transformation loss 
#             - recall   registration recall (re < re_thre & te < te_thre)
#             - RE       rotation error 
#             - TE       translation error
#             - RMSE     RMSE under the predicted transformation
#         """
#         bs = trans.shape[0]
#         R, t = decompose_trans(trans)
#         gt_R, gt_t = decompose_trans(gt_trans)

#         recall = 0
#         RE = torch.tensor(0.0).to(trans.device)
#         TE = torch.tensor(0.0).to(trans.device)
#         RMSE = torch.tensor(0.0).to(trans.device)
#         loss = torch.tensor(0.0).to(trans.device)
#         for i in range(bs):
#             re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
#             te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
#             warp_src_keypts = transform(src_keypts[i], trans[i])
#             rmse = torch.norm(warp_src_keypts - tgt_keypts, dim=-1).mean()
#             re = re * 180 / np.pi
#             te = te * 100
#             if te < self.te_thre and re < self.re_thre:
#                 recall += 1
#             RE += re
#             TE += te
#             RMSE += rmse

#             pred_inliers = torch.where(probs[i] > 0)[0]
#             if len(pred_inliers) < 1:
#                 loss += torch.tensor(0.0).to(trans.device)
#             else:
#                 warp_src_keypts = transform(src_keypts[i], trans[i])
#                 loss +=  ((warp_src_keypts - tgt_keypts)**2).sum(-1).mean()

#         return loss / bs, recall * 100.0 / bs, RE / bs, TE / bs, RMSE / bs


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(torch.nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=64, device='cuda:0'):
        super().__init__()
        self.hidden_num_feature = hidden_num_feature
        self.conv1 = PointNetLayer(in_num_feature, self.hidden_num_feature)
        self.conv2 = PointNetLayer(self.hidden_num_feature, output_num_feature)
        # self.conv3 = PointNetLayer(2*self.hidden_num_feature, output_num_feature)
        self.device = device
        
    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        print("&&&&&&&&&&&&")
        print(h.shape)
        # Global Pooling:
        #h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        self.to(self.device)
        return h
    
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        print('$$$$lora hhhhhhh$$$$')
        print(x.shape)
        print(self.A.shape)
        print(self.B.shape)
        x = self.alpha * (x @ self.A @ self.B)
        return x

class crossAttentionRegression(torch.nn.Module):
    def __init__(self, output_dim, eps=1e-6):
        super().__init__()
        self.output_dim = output_dim
        self.eps = eps
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.regression = None
        self.head = None
        # self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)        

    def forward(self, src_tensor, tar_tensor, batch_size = 1):
        # print('$$$$lora hhhhhhh$$$$')
        # print(self.src_tensor.shape)
        # print(self.tar_tensor.shape)
        # attent_weghts = self.cos(src_tensor, tar_tensor)
        """
        added eps for numerical stability
        """
        a_n, b_n = src_tensor.norm(dim=1)[:, None], tar_tensor.norm(dim=1)[:, None]
        a_norm = src_tensor / torch.clamp(a_n, min=self.eps)
        b_norm = tar_tensor / torch.clamp(b_n, min=self.eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        print("########attention weights")
        print(sim_mt.shape)
        print(b_norm.shape)
        print(a_norm.shape)
        proj_src = sim_mt @ b_norm
        proj_tar = sim_mt.transpose(0, 1) @ a_norm

        perm_proj_src = torch.permute(proj_src, (1, 0)) #########batch size not added
        perm_proj_tar = torch.permute(proj_tar, (1, 0)) #########batch size not added
        src0 = self.max_pool(perm_proj_src).view(batch_size, -1)
        print(src0.shape)
        src1 = self.avg_pool(perm_proj_tar).view(batch_size, -1)
        x = torch.cat((src0, src1), dim=-1)

        return sim_mt, proj_src, proj_tar


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer, basic moduel to buil up the eqnn network model
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, device='cuda:0'):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.device =device

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), ########2*input node feature, edge feature, edge attribute dimension features
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=1e-3)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused for edge attributes.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        if edge_feat is not None:
            trans = coord_diff * self.coord_mlp(edge_feat)
        trans = coord_diff
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        self.to(self.device)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_lora_dim, out_lora_dim, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=1, residual=True, attention=False, normalize=False, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.out_lora_dim = out_lora_dim
        self.in_lora_dim = in_lora_dim
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.self_attention = None
        self.LoRAlayer = LoRALayer(self.in_lora_dim, self.out_lora_dim, out_node_nf+3) ###in_node_nf 64 

    def forward(self, h, x, edges, edge_attr):
        # self.to(self.device)
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        concat_feats = torch.cat([x, h], dim=1)
        print('^^^^^')
        print(x.shape)
        print(h.shape)
        print(concat_feats.shape)
        loRA_feats = self.LoRAlayer(concat_feats)
        return h, x, loRA_feats


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1).cuda()
    edges = [torch.LongTensor(edges[0]).cuda(), torch.LongTensor(edges[1]).cuda()]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows).cuda(), torch.cat(cols).cuda()]
    return edges, edge_attr


if __name__ == "__main__":
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_pos = torch.rand(2048, 3).to(dev)
    src_edge = torch.randint(1, 10, (2, 768)).to(dev)
    batch = torch.tensor([1]).to(dev)

    tar_pos = torch.rand(2048, 3).to(dev)
    tar_edge = torch.randint(1, 10, (2, 768)).to(dev)
#   dataset.transform = T.Compose([SamplePoints(num=256), KNNGraph(k=6)])

#   data = dataset[0]
#   y = [1, 3]
#   Data(pos, edge_index=[2, 1536], y)
    k = 12
    src_graph_idx = knn_graph(src_pos,k,loop=False)
    src_index = torch.LongTensor([1, 0]) #####swap index
    src_new_graph_idx = torch.zeros_like(src_graph_idx)
    src_new_graph_idx[src_index] = src_graph_idx
    src_ks_graph_coords = src_pos[src_new_graph_idx]
    # src_mask = torch.logical_xor(src_new_graph_idx[0,:], src_new_graph_idx[1,:]) 


    tar_graph_idx = knn_graph(tar_pos,k,loop=False)
    tar_index = torch.LongTensor([1, 0]) #####swap index
    tar_new_graph_idx = torch.zeros_like(tar_graph_idx)
    tar_new_graph_idx[tar_index] = tar_graph_idx
    tar_ks_graph_coords = src_pos[tar_new_graph_idx]
    # src_mask = torch.logical_xor(src_new_graph_idx[0,:], src_new_graph_idx[1,:]) 


    # edgegraph = KNNGraph(k=12) #RadiusGraph(r=0.05, loop = False, max_num_neighbors = 15, num_workers = 6)
    print(src_graph_idx.shape)
    print(src_pos.shape)
    print(src_ks_graph_coords.shape)
    # print(pos)
    # print(ks_graph)
    feature_encoder1 = PointNet().to(dev)
    feats1 = feature_encoder1(src_pos, src_edge, batch).to(dev)

    print(feats1.shape)
    # Dummy parameters
    n_nodes1 = torch.tensor([src_pos.shape[0]]).to(dev)
    n_feat1 =  torch.tensor([feats1.shape[1]]).to(dev)
    x_dim1 = torch.tensor([src_pos.shape[1]]).to(dev)

    n_nodes2 = torch.tensor([src_pos.shape[0]]).to(dev)
    n_feat2 =  torch.tensor([feats1.shape[1]]).to(dev)
    x_dim2 = torch.tensor([src_pos.shape[1]]).to(dev)

    # Dummy variables h, x and fully connected edges
    h1 = torch.ones(batch * n_nodes1, n_feat1).to(dev)
    x1 = torch.ones(batch * n_nodes1, x_dim1).to(dev)
    edges1, edge_attr1 = get_edges_batch(n_nodes1, batch)

    # Dummy variables h, x and fully connected edges
    h2 = torch.ones(batch * n_nodes2, n_feat2).to(dev)
    x2 = torch.ones(batch * n_nodes2, x_dim2).to(dev)
    edges2, edge_attr2 = get_edges_batch(n_nodes2, batch)

    print('^^^^^^^###########^^^^^^^^')
    print(edge_attr1.shape)

    # # Initialize EGNN
    # egnn = EGNN(in_node_nf=n_feat1, hidden_nf=32, out_node_nf=1, in_lora_dim = src_pos.shape[1]+1, out_lora_dim = 6,in_edge_nf=1).cuda()

    # # Run EGNN
    # h1, x1, reduc_feats1 = egnn(h1, x1, edges1, edge_attr1)
    # h2, x2, reduc_feats2 = egnn(h2, x2, edges2, edge_attr2)

    # print('#######h1  x1#######')
    # print(h1)
    # print(x1)
    # cos = crossAttentionRegression(output_dim=4, eps=1e-6)
    # output, _, _ = cos(reduc_feats1, reduc_feats2, batch)
    # print(model)
