
import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
import wandb

torch.set_default_dtype(torch.float64) # works best in float64

option = "atten"
 
base_dir = '/home/eavise3d/Downloads/rgbd_dataset_freiburg1_xyz'

filename = os.path.join(base_dir, 'result.pth')  #####data load, read in
# data = {}
print('############')
data = torch.load(filename)
# print(data)
src_frame_fea = data['feats0_results']['keypoints']
src_frame_descriptor= data['feats0_results']['descriptors']
src_frame_points_scores = data['feats0_results']['keypoint_scores']
#################################################################
tar_frame_fea = data['feats1_results']['keypoints']
tar_frame_descriptor = data['feats1_results']['descriptors']
tar_frame_points_scores = data['feats1_results']['keypoint_scores']

print(data['feats0_results']['keypoints'].shape)
print(data['matches01']['matches1'].shape)
print(data['feats1_results']['keypoints'].shape)
print(src_frame_points_scores.shape)

# geom = torch.tensor(points.transpose())[None,:].double()
# feat = torch.randint(0, 20, (1, geom.shape[1],1)).double()

feats1 = torch.unsqueeze(src_frame_descriptor, 0).cuda() #torch.randn(2, 32, 32).cuda()
coors1 = torch.unsqueeze(src_frame_fea, 0).cuda() #torch.randn(2, 32, 3).cuda()
#rel_pos1 = torch.randint(0, 4, (2, 32, 32)).cuda()
# mask1  = src_frame_points_scores.bool().cuda()
mask1  = torch.ones(1, 2048).bool()

feats2 = tar_frame_descriptor.cuda() #torch.randn(2, 32, 32).cuda()
coors2 = tar_frame_fea.cuda() #torch.randn(2, 32, 3).cuda()
#rel_pos2 = torch.randint(0, 4, (2, 32, 32)).cuda()
mask2  = tar_frame_points_scores.bool().cuda()

# atom_feats = torch.randn(2, 32, 64)
# coors = torch.randn(2, 32, 3)
# mask  = torch.ones(2, 32).bool()
print(feats1.shape[0])
print(coors1.shape[0])
# print(data['matches01'])
print('$$$$$$$$$$$$$$$$$$$$')
print(data['matches01']['matches0'].shape)
print(data['matches01']['matches1'].shape)
print(data['matches01']['matching_scores0'].shape)
print(data['matches01']['matching_scores1'].shape)
print(data['matches01']['matches'].shape)

print(data['matches01']['scores'].shape)

pair_idxs = data['matches01']['matches'].cuda()
match_scores = data['matches01']['scores'].cuda()

matched_src_frame_fea = src_frame_fea[pair_idxs[:, 0]].cuda()
matched_src_frame_descriptor = src_frame_descriptor[pair_idxs[:, 0]].cuda()
matched_src_frame_points_scores = src_frame_points_scores[pair_idxs[:, 0]].cuda() #torch.zeros_like(pair_idxs[:, 0])

matched_tar_frame_fea = tar_frame_fea[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])
matched_tar_frame_descriptor = tar_frame_descriptor[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])
matched_tar_frame_points_scores = tar_frame_points_scores[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])


print('&&&&&&&&&&')
print(matched_src_frame_descriptor.shape)
print(matched_src_frame_points_scores.shape)


# print(matched_src_frame_fea.shape)

# for i, item in enumerate(pair_idxs):
#     matched_src_frame_fea.append(src_frame_fea[item[0]])
#     matched_src_frame_descriptor.append(src_frame_descriptor[item[0]])
#     matched_src_frame_points_scores.append(src_frame_points_scores[item[0]])

#     matched_tar_frame_fea.append(tar_frame_fea[item[1]])
#     matched_tar_frame_descriptor.append(tar_frame_fea[item[1]])
#     matched_tar_frame_points_scores.append(tar_frame_points_scores[item[1]])

refined_coors1 = torch.zeros_like(coors1).cuda()
refined_coors2 = torch.zeros_like(coors1).cuda()

print(matched_src_frame_fea.shape)
print(matched_src_frame_descriptor.shape)
print(matched_src_frame_points_scores.shape)

if option == "atten":
    print('&&&&&&&&&&&hahahhahaha&&&1&&&&&')
    
    encoder = SE3Transformer(
        # dim_in = src_frame_fea.shape[1],
        # dim = 512,
        # heads = 8,
        # depth = 3,
        # dim_head = 64,
        # num_degrees = 4,
        # valid_radius = 10

        dim = 1,
        heads = 4,
        depth = 1,
        dim_head = 1,
        num_degrees = 1 ,
        valid_radius = 5
        # valid_radius = 10
        # differentiable_coors = True
    ).cuda()
    print('&&&&&&&&&&&hahahhahaha&&2&&&&&&')

    feats = torch.randn(1, 100, 1).cuda()
    coors = torch.randn(1, 100, 3).cuda()
    mask  = torch.ones(1, 100).bool().cuda()

    out = encoder(feats, coors, mask) # (1, 1024, 512)

    # refined_coors1 = coors1 + encoder(feats1, coors1, return_type = 1) # (2, 32, 3)  "
    # refined_coors2 = coors2 + encoder(feats2, coors2, mask2, return_type = 1) # (2, 32, 3)  "
    print('&&&&&&&&&&&hahahhahaha&&&3&&&&&')
    print(refined_coors1.shape)

elif option == "egnn":
    encoder = SE3Transformer(
        dim = src_frame_fea.shape[1],
        num_neighbors = 5,
        # valid_radius = 10,
        # num_edge_tokens = 4,
        # edge_dim = 4,
        num_degrees = 4,       # number of higher order types - will use basis on a TCN to project to these dimensions
        use_egnn = True,       # set this to true to use EGNN instead of equivariant attention layers
        egnn_hidden_dim = 64,  # egnn hidden dimension
        depth = 4,             # depth of EGNN
        reduce_dim_out = True  # will project the dimension of the higher types to 1
    ).cuda()

    refined_coors1 = coors1 + encoder(feats1, coors1, mask1, return_type = 1) # (2, 32, 3) edges = rel_pos1, 
    refined_coors2 = coors2 + encoder(feats2, coors2, mask2, return_type = 1) # (2, 32, 3)  "

print('feats **************')
#print(refinement)
  
  # update coors with refinement    

# class Trainer(object):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.type = self.config['General']['type']

#         self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        
#         print("##########device: %s" % self.device)
#         resize = config['Dataset']['transforms']['resize']
#         # self.model = ConvFocus(config)
#         self.model = None

#         self.model.to(self.device)
#         # print(self.model)
#         # exit(0)

#         self.loss_depth = get_losses(config)
#         self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
#         self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])
    

#     def train(self, train_dataloader, val_dataloader):
#         epochs = self.config['General']['epochs']
#         if self.config['wandb']['enable']:
#             wandb.init(project="FocusOnDepth", entity=self.config['wandb']['username'])
#             wandb.config = {
#                 "learning_rate_backbone": self.config['General']['lr_backbone'],
#                 "epochs": epochs,
#                 "batch_size": self.config['General']['batch_size']
#             }
#         val_loss = Inf
#         # summary(self.model, (-1, 3, 384, 384))
#         print(self.model)

#         for epoch in range(epochs):  # loop over the dataset multiple times
#             print("Epoch ", epoch+1)
#             running_loss = 0.0
#             self.model.train()
#             pbar = tqdm(train_dataloader)
#             pbar.set_description("Training")
#             for i, (X,  Y_depths) in enumerate(pbar):
#                 # get the inputs; data is a list of [inputs, labels]
#                 X, Y_depths = X.to(self.device), Y_depths.to(self.device)
#                 # zero the parameter gradients
#                 self.optimizer_backbone.zero_grad()
#                 output_depths = self.model(X, hidden)
#                 output_depths = output_depths.squeeze(1) if output_depths != None else None

#                 Y_depths = Y_depths.squeeze(1) #1xHxW -> HxW
#                 # get loss
#                 loss = self.loss_depth(output_depths, Y_depths)
#                 loss.backward()
#                 # step optimizer

#                 self.optimizer_scratch.step()
#                 self.optimizer_backbone.step()

#                 running_loss += loss.item()
#                 if np.isnan(running_loss):
#                     print('\n',
#                         X.min().item(), X.max().item(),'\n',
#                         Y_depths.min().item(), Y_depths.max().item(),'\n',
#                         output_depths.min().item(), output_depths.max().item(),'\n',
#                         loss.item(),
#                     )
#                     exit(0)

#                 # writer.add_scalar('loss', running_loss, epoch)
#                 if self.config['wandb']['enable'] and ((i % 50 == 0 and i>0) or i==len(train_dataloader)-1):
#                     wandb.log({"loss": running_loss/(i+1)})
#                 pbar.set_postfix({'training_loss': running_loss/(i+1)})

#             new_val_loss = self.run_eval(val_dataloader)

#             if epoch%25 ==0:
#                 self.save_model()

#             if new_val_loss < val_loss:
#                 val_loss = new_val_loss
#                 self.save_model()
#             # writer.add_scalar('val_loss', new_val_loss, epoch)

#             self.schedulers[0].step(new_val_loss)

#         print('Finished Training')

#     def run_eval(self, val_dataloader):
#         """
#             Evaluate the model on the validation set and visualize some results
#             on wandb
#             :- val_dataloader -: torch dataloader
#         """
#         val_loss = 0.
#         self.model.eval()
#         X_1 = None
#         Y_depths_1 = None
#         #Y_segmentations_1 = None
#         output_depths_1 = None
#         #output_segmentations_1 = None
#         with torch.no_grad():
#             pbar = tqdm(val_dataloader)
#             pbar.set_description("Validation")
#             for i, (X, Y_depths) in enumerate(pbar):
#                 X, Y_depths = X.to(self.device), Y_depths.to(self.device)
#                 # hidden = self.model.init_hidden(hidden_layer_num)
#                 output_depths, hidden = self.model(X, hidden)
#                 output_depths = output_depths.squeeze(1) if output_depths != None else None
#                 Y_depths = Y_depths.squeeze(1)
#                 #Y_segmentations = Y_segmentations.squeeze(1)
#                 if i==0:
#                     X_1 = X
#                     Y_depths_1 = Y_depths
#                     #Y_segmentations_1 = Y_segmentations
#                     output_depths_1 = output_depths
#                     #output_segmentations_1 = output_segmentations
#                 # get loss
#                 loss = self.loss_depth(output_depths, Y_depths)
#                 val_loss += loss.item()
#                 pbar.set_postfix({'validation_loss': val_loss/(i+1)})
#             if self.config['wandb']['enable']:
#                 wandb.log({"val_loss": val_loss/(i+1)})
#                 self.img_logger(X_1, Y_depths_1, output_depths_1)
#         return val_loss/(i+1)


def save_model(self):
    path_model = os.path.join(self.config['General']['path_model'], self.model.__class__.__name__)
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



