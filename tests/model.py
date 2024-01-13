
import torch
from torch import nn
import torch.optim as optim
# from egnn_pytorch import EGNN
# from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
import wandb
import json
#from sklearn.neighbors import NearestNeighbors
from datasets import TUMDataset

torch.cuda.manual_seed(2)
torch.set_default_dtype(torch.float64) # works best in float64

option = "egnn"

# if option == "atten":
#     print('&&&&&&&&&&&hahahhahaha&&&1&&&&&')
    
#     # encoder = SE3Transformer(
#     #     # dim_in = src_frame_fea.shape[1],
#     #     dim = 64,
#     #     heads = 4,
#     #     depth = 1,
#     #     dim_head = 1,
#     #     num_degrees = 1 ,
#     #     valid_radius = 8,
#     #     reduce_dim_out = True,
#     # ).cuda()

#     transformer = SE3Transformer(
#         dim = 6,
#         heads = 2,
#         dim_head = 8,
#         depth = 1,
#         valid_radius = 8,
#         input_degrees = 1, # 2 will keep feature dim
#         num_degrees = 2,
#         output_degrees = 2,
#         reduce_dim_out = True,   ####if fales is b, num, fea_dim,  coordinate dim.
#     ).cuda()

#     print('&&&&&&&&&&&hahahhahaha&&2&&&&&&')
#     feats = torch.randn(1, 100, 6).cuda()
#     coors = torch.randn(1, 100, 3).cuda()
#     mask  = torch.ones(1, 100).bool().cuda()
#     print('@@@@@@@@@@@@@@')
#     print(feats.shape)
#     print(coors.shape)
#     print(mask.shape)


#     # out = encoder(feats, coors, mask,return_type = 1) # (1, 1024, 512)
#     # print(out.shape) 

#     out2 = transformer(feats, coors, mask,return_type = 1) # (1, 1024, 512)
#     print(out2.shape)
#     # print(out - coors1)
#     # refined_coors1 = coors1 + encoder(feats1, coors1, return_type = 1) # (2, 32, 3)  "
#     # refined_coors2 = coors2 + encoder(feats2, coors2, mask2, return_type = 1) # (2, 32, 3)  "
#     print('&&&&&&&&&&&hahahhahaha&&&3&&&&&')
#     print(refined_coors1.shape)

# elif option == "egnn":
#     encoder = SE3Transformer(
#         dim = 6, #src_frame_fea.shape[1],
#         # valid_radius = 10,
#         num_neighbors = 8,
#         num_edge_tokens = 4,
#         edge_dim = 4,
#         num_degrees = 1,       # number of higher order types - will use basis on a TCN to project to these dimensions
#         use_egnn = True,       # set this to true to use EGNN instead of equivariant attention layers
#         egnn_hidden_dim = 6,  # egnn hidden dimension
#         depth = 1,             # depth of EGNN
#         reduce_dim_out = True  # will project the dimension of the higher types to 1    
    
#     ).cuda()

#     feats = torch.randn(1, 50, 6).cuda()
#     coors = torch.randn(1, 50, 3).cuda()
#     bonds = torch.randint(0, 4, (1, 50, 50)).cuda()
#     mask  = torch.ones(1, 50).bool().cuda()
#     print('&&&&&&&&&&&hahahhahaha&&&3&&&&&')
#     refinement = encoder(feats, coors, mask, edges = bonds, return_type = 1)
#     coors = coors + refinement
#     print(coors)
#     refined_coors1 = coors1 + encoder(feats1, coors1, mask1, return_type = 1) # (2, 32, 3) edges = rel_pos1, 
#     refined_coors2 = coors2 + encoder(feats2, coors2, mask2, return_type = 1) # (2, 32, 3)  "

#     update coors with refinement    


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
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

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
        if edge_attr is None:  # Unused.
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
        trans = coord_diff * self.coord_mlp(edge_feat)
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
    
    # def gen_rnn_local_graph(
    #     points_xyz, center_xyz, radius, num_neighbors,
    #     neighbors_downsample_method='random',
    #     scale=None):
    #     """Generate a local graph by radius neighbors.
    #     """
    #     if scale is not None:
    #         scale = np.array(scale)
    #         points_xyz = points_xyz/scale
    #         center_xyz = center_xyz/scale
    #     nbrs = NearestNeighbors(
    #         radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    #     indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
    #     if num_neighbors > 0:
    #         if neighbors_downsample_method == 'random':
    #             indices = [neighbors if neighbors.size <= num_neighbors else
    #                 np.random.choice(neighbors, num_neighbors, replace=False)
    #                 for neighbors in indices]
    #     vertices_v = np.concatenate(indices)
    #     vertices_i = np.concatenate(
    #         [i*np.ones(neighbors.size, dtype=np.int32)
    #             for i, neighbors in enumerate(indices)])
    #     vertices = np.array([vertices_v, vertices_i]).transpose()
    #     return vertices

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=1, residual=True, attention=False, normalize=False, tanh=False):
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
        #####       we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
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
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


def save_model(self, base_dir, filename):
    path_model = os.path.join(base_dir, filename)
    try:
        os.makedirs(path_model)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    torch.save({'model_state_dict': self.model.state_dict(),
                'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                }, path_model+'.pth')
    print('Model saved at : {}'.format(path_model))


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}
    device = 'cuda:0'
    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, edge_attr, loc_end = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()


        if args.model == 'egnn':
            nodes = torch.ones(loc.size(0), 1).to(device)  # all input nodes are set to 1
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist, vel_attr], 1).detach()  # concatenate all edge properties

            loc_pred = model(nodes, loc.detach(), edges, edge_attr)
        elif args.model == 'se3_transformer' or args.model == 'tfn':
            loc_pred = model(loc, feats)

        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
        loss = loss_mse(loc_pred, loc_end)
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        if batch_idx % args.log_interval == 0 and (args.model == "se3_transformer" or args.model == "tfn"):
            print('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    # Dummy parameters
    model = "egnn"
    batch_size = 1
    n_nodes = 100
    n_feat = 6
    x_dim = 3
    epochs = 1000
    test_interval = 50
    max_training_samples = 1024

    dataset = "TUMRGBD"
    out_dir = "/home/eavise3d/Downloads/match/output"
    exp_name = ""
    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN from the library
    # h, x = egnn(h, x, edges, edge_attr)

    # dataset_train = TUMDataset(partition='train', dataset_name=dataset,
    #                              max_samples=max_training_samples)
    # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)

    # dataset_val = TUMDataset(partition='val', dataset_name="nbody_small")
    # loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    # dataset_test = TUMDataset(partition='test', dataset_name="nbody_small")
    # loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)



    if model == 'egnn':
        egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)
        print(egnn)

    # elif model == 'se3_transformer':
    #     tfn = SE3Transformer(n_particles=5, n_dimesnion=3, nf=int(nf/degree), n_layers=n_layers, model=model, num_degrees=degree, div=1)
    #     if torch.cuda.is_available():
    #         tfn = tfn.cuda()
    else:
        raise Exception("Wrong model specified")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # results = {'epochs': [], 'losess': []}
    # best_val_loss = 1e8
    # best_test_loss = 1e8
    # best_epoch = 0
    # for epoch in range(0, epochs):
    #     train(model, optimizer, epoch, loader_train)
    #     if epoch % test_interval == 0:
    #         #train(epoch, loader_train, backprop=False)
    #         val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
    #         test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
    #         results['epochs'].append(epoch)
    #         results['losess'].append(test_loss)
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_test_loss = test_loss
    #             best_epoch = epoch
    #         print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))

    #     json_object = json.dumps(results, indent=4)
    #     with open(out_dir + "/" + exp_name + "/losess.json", "w") as outfile:
    #         outfile.write(json_object)
    # print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))

 



