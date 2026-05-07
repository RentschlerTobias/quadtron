
import numpy as np
import matplotlib.pyplot as plt
import plotting_tools
from trainer import Trainer
import torch
from tokenizer_v2 import Tokenizer2D as tokenizer_v2
from tokenizer import Tokenizer2D as tokenizer_v1
from tqdm import tqdm
from collections import defaultdict
from dataset import MeshData
import os

import torch_geometric

from torch_geometric.data import Data

# path_meshes1 = '../data/structured_quad_meshes_pre_selected_v2.pt'
# path_meshes = '../data/new_checkpoints/checkpoint_mesh_720.pt'
# folder = '../data/new_checkpoints/'
# all_tensors = []
#
# for i, file in enumerate(os.listdir(folder)):
#     if file.endswith(".pt"):
#         print(file)
#         tensor = torch.load(os.path.join(folder, file), weights_only=False)
#         for t in tensor:
#             all_tensors.append(t)
#
# len(all_tensors)
#
# torch.save(all_tensors, './centered_blades.pt')
#
# meshes = torch.load('./centered_blades.pt', weights_only=False)
# mesh = meshes1[0]
# new_meshes = []
# len(meshes)
# meshes = torch.load(path_meshes, weights_only=False)
#
# for mesh in meshes:
#     if mesh.quad_faces.size(1) > 55:
#         continue
#     m = Data(x=mesh.quad_coordinates.clone(), faces=mesh.quad_faces.clone(
#     ), tri_coordinates=mesh.tri_coordinates.clone())
#     new_meshes.append(m)
# len(new_meshes)
# torch.save(new_meshes, './centered_blades.pt')
#
# for i in range(20):
#     print(i)
#     print(new_meshes[i].faces.size())
#
#
# mesh = meshes[0]
#
# plotting_tools.plt_mesh(mesh.x, mesh.faces)
#
#
path_meshes = './centered_blades.pt'
quantization = 1024
d_model = 512
n_latents = 2*d_model
batch_size = 12
num_epochs = 50
learning_rate = 1e-4
stage_layers = [2, 2, 4, 2, 2]
window_size = None
n_heads = 4
verbose = False
verbose = False

sorting_strategy = 0
epoch_saving_point = 39
gradient_accumulation = None

trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                  gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose, sorting_strategy=sorting_strategy)


# path = f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}_batch_size_{batch_size}_n_heads_{n_heads}_window_size_{window_size}_sorting_strategy_{sorting_strategy}_stage_layers_' +
# '_'.join(str(s) for s in stage_layers)+f'/epoch_{epoch_saving_point}.pt'

path = (
    f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}_batch_size_{batch_size}_n_heads_{
        n_heads}_window_size_{window_size}_sorting_strategy_{sorting_strategy}_stage_layers_'
    + '_'.join(str(s) for s in stage_layers)
    + f'/epoch_{epoch_saving_point}.pt'
)
# path = f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}_batch_size_{batch_size}_n_heads_{n_heads}_window_size_{window_size}_stage_layers_' + \
#     '_'.join(str(s) for s in stage_layers)+f'/epoch_{epoch_saving_point}.pt'
#

checkpoint = torch.load(path, weights_only=True)

min(checkpoint['training_history']['val_loss'])

state = checkpoint['model_state_dict']

trainer.model.load_state_dict(state)


#
#
# meta_mesh = torch.load('./meta_mesh.pt', weights_only=False)
#
# mesh0=meta_mesh.clone()
# mesh0.x=meta_mesh.nodes_T3
# mesh0.faces  =  torch.tensor(0)
# mesh0.faces=meta_mesh.faces_T3
#
# mesh1=meta_mesh.clone()
# mesh1.x=meta_mesh.nodes_T4
# mesh1.faces  =  torch.tensor(0)
# mesh1.faces=meta_mesh.faces_T4
#
#
# mesh0.faces.size()
# mesh1.faces.size()
#
# meshes =[mesh0,mesh1]
#
# tokenizer = tokenizer_v2(quantization_levels=quantization, verbose=True, sorting_strategy=sorting_strategy)
# test_data = MeshData(meshes, tokenizer, verbose=True)
#
# test_data.face_count
# test_data.data[1]
# test_data.point_clouds[0].size()
# points0 =test_data.point_clouds[0].unsqueeze(0)
# points0.size()
# faces0=torch.tensor([test_data.face_count[0]])
#
# test_data.point_clouds[1].size()
# points1 =test_data.point_clouds[1].unsqueeze(0)
# points1.size()
# faces1=torch.tensor([test_data.face_count[1]])
#
# point_cloud.size()
# face_count=faces0
# point_cloud = points0
#
# face_count=faces1
# point_cloud = points1
#
counter = 0

data = next(iter(trainer.val_loader))
input_tokens = data['input_tokens'][0:1]
target_tokens = data['target_tokens'][0:1]
point_cloud = data['point_cloud'][0:1]
face_count = data['face_count'][0:1]
start = input_tokens[0:1, 0:8]
print(face_count)
# face_count[0] = 54
generated_tokens = trainer.model.generate(
    point_cloud=point_cloud,
    face_count=face_count,
    start_tokens=start,
    max_length=500
)


vertices, quads = trainer.tokenizer.detokenize(generated_tokens)
plotting_tools.plt_mesh(
    vertices, quads, point_cloud=point_cloud[0], output_file=f'./figures/generated_meshes/mesh_{counter}.png')
counter += 1


quads.size()

vertices_true, quads_true = trainer.tokenizer.detokenize(input_tokens[0])
plotting_tools.plt_mesh(vertices_true, quads_true,
                        output_file='./figures/true_mesh.png')

plotting_tools.plt_point_cloud(point_cloud[0])


plotting_tools.plt_mesh(mesh.x[:, 0:2], mesh.faces,
                        output_file='./true_mesh.png')

plotting_tools.plt_mesh(vertices, quads, point_cloud=points,
                        output_file='./test_mesh.png')


sorting_strategy = 0
tokenizer_1 = tokenizer_v1(quantization_levels=quantization, verbose=verbose)
tokenizer_2 = tokenizer_v2(quantization_levels=quantization,
                           verbose=verbose, sorting_strategy=sorting_strategy)

index = 2
mesh = meshes[index]

plotting_tools.plt_mesh(mesh.x[:, 0:2], mesh.faces, './figures/true_mesh.png')

plotting_tools.plt_point_cloud(
    mesh.x[:, 0:2], mesh.faces, './figures/true_mesh.png')


tokens_2 = tokenizer_2.tokenize(mesh.x[:, 0:2], mesh.faces)
vertices_2, quads_2 = tokenizer_2.detokenize(tokens_2)
plotting_tools.plt_mesh(
    vertices_2, quads_2, f'./figures/tokenizer_v2_sorting_{sorting_strategy}.png')


tokens_1 = tokenizer_1.tokenize(mesh.x[:, 0:2], mesh.faces)
vertices_1, quads_1 = tokenizer_1.detokenize(tokens_1)
plotting_tools.plt_mesh(vertices_1, quads_1, './figures/tokenizer_v1.png')
