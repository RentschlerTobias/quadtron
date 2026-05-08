import torch
import plotting_tools
from trainer import Trainer
from tokenizer_v2 import Tokenizer2D
from dataset import MeshData
from torch_geometric.data import Data

# --- Config (muss mit Checkpoint übereinstimmen) ---
path_meshes = './centered_blades.pt'
quantization = 1024
d_model = 512
n_latents = 2 * d_model
batch_size = 4
num_epochs = 50
learning_rate = 1e-4
stage_layers = [2, 2, 4, 2, 2]
window_size = 200
n_heads = 4
sorting_strategy = 0
epoch_saving_point = 23
gradient_accumulation = None

# --- Meta-Mesh laden (zwei Verfeinerungen derselben Geometrie) ---
meta_mesh = torch.load('./meta_mesh.pt', weights_only=False)

mesh0 = Data(
    x=meta_mesh.nodes_T3.clone(),
    faces=meta_mesh.faces_T3.clone(),
    tri_coordinates=meta_mesh.tri_coordinates.clone()
)
mesh1 = Data(
    x=meta_mesh.nodes_T4.clone(),
    faces=meta_mesh.faces_T4.clone(),
    tri_coordinates=meta_mesh.tri_coordinates.clone()
)

meshes = [mesh0, mesh1]
tokenizer = Tokenizer2D(quantization_levels=quantization,
                        sorting_strategy=sorting_strategy)
test_data = MeshData(meshes, tokenizer, verbose=True)

for i, mesh in enumerate(meshes):
    face_count = test_data.face_count[i]
    print(face_count)
    point_cloud = test_data.point_clouds[i].unsqueeze(0)
    face_count_tensor = torch.tensor([face_count])

    print(f"\n=== Mesh {i}: {face_count} faces ===")

    vertices_true = test_data.meshes[i].x[:, 0:2].clone()
    quads_true = test_data.meshes[i].faces.clone()

    # true_tokens = tokenizer.tokenize(mesh.x[:, 0:2], mesh.faces)
    # vertices_true, quads_true = tokenizer.detokenize(true_tokens)
    plotting_tools.plt_mesh(vertices_true, quads_true,
                            output_file=f'./figures/true_mesh_{i}_faces{face_count}.png')
    print(f"  True mesh: {
          quads_true.shape[1] if quads_true.dim() > 1 else 0} faces geplottet")
