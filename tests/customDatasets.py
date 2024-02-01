import torch
from torch import nn
import torch.optim as optim
# from egnn_pytorch import EGNN
# from se3_transformer_pytorch import SE3Transformer
import egcnModel
from gcnLayer import GraphConvolution, GlobalPooling
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
import wandb
import json
import os.path as osp
import os


class Match3DDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, **kwargs):
        super(Match3DDataset, self).__init__(root)
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        processed_dir = osp.join(root, "processed")
        #print(self.datalist)
        self.split = split

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
        self.datalist = ['data_{}_{}.pt'.format(i, self.split) for i in range(len(self.links))]
        
        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
    
    @property
    def processed_file_names(self):
        return self.datalist
 
    def __len__(self):
        return len(self.links)
    def _process(self):
        os.makedirs(self.processed_dir)
        if len(glob.glob(osp.join(self.processed_dir, '*.pt'))) > 0:
            return
        self.process()

    def process(self)        
        for idx in tqdm(range(len(self.links))):
            src, dst = self.links[idx]

            if self.labels[idx]: status = "pos"
            else: status = "neg"

            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, status, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.data.x)
            data = construct_pyg_graph(*tmp, self.node_label)

            torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
            self.datalist.append('data_{}_{}.pt'.format(idx, self.split))
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
        return data
    
    

    def _process(self):
        os.makedirs(self.processed_dir)
        if len(glob.glob(osp.join(self.processed_dir, '*.pt'))) > 0:
            return
        self.process()

    def process(self)
        for idx in tqdm(range(len(self.links))):
            src, dst = self.links[idx]

            if self.labels[idx]: status = "pos"
            else: status = "neg"

            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, status, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.data.x)
            data = construct_pyg_graph(*tmp, self.node_label)

            torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
            self.datalist.append('data_{}_{}.pt'.format(idx, self.split))