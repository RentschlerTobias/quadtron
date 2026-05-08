import torch
import plotting_tools
from trainer import Trainer
from tokenizer_v2 import Tokenizer2D
from dataset import MeshData
from torch_geometric.data import Data

# --- Config (muss mit Checkpoint übereinstimmen) ---
path_meshes = './centered_blades_cleaned.pt'
quantization = 1024
d_model = 512
n_latents = 2 * d_model
batch_size = 8
num_epochs = 50
learning_rate = 1e-4
stage_layers = [2, 2, 2, 2, 2]
window_size = 200
n_heads = 4
sorting_strategy = 5
epoch_saving_point = 27
gradient_accumulation = None

# --- Modell laden ---
trainer = Trainer(
    data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate,
    batch_size=batch_size, quantization=quantization, d_model=d_model,
    n_latents=n_latents, gradient_accumulation=gradient_accumulation,
    window_size=window_size, n_heads=n_heads, stage_layers=stage_layers,
    verbose=False, sorting_strategy=sorting_strategy
)

checkpoint_path = (
    f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}'
    f'_batch_size_{batch_size}_n_heads_{n_heads}_window_size_{window_size}'
    f'_sorting_strategy_{sorting_strategy}_stage_layers_'
    + '_'.join(str(s) for s in stage_layers)
    + f'/epoch_{epoch_saving_point}.pt'
)

checkpoint = torch.load(checkpoint_path, weights_only=True)
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.model.eval()
print(f"Checkpoint geladen. Bestes val_loss: {
      min(checkpoint['training_history']['val_loss']):.4f}")

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

for mesh in meshes:
    print(mesh.faces.size(1))

# --- Für jedes Mesh: true + generated plotten ---
for i, mesh in enumerate(meshes):
    mesh = meshes[i].clone()
    face_count = test_data.face_count[i]
    point_cloud = test_data.point_clouds[i].unsqueeze(0)
    face_count_tensor = torch.tensor([face_count])

    print(f"\n=== Mesh {i}: {face_count} faces ===")

    true_tokens = tokenizer.tokenize(mesh.x[:, 0:2], mesh.faces)
    vertices_true, quads_true = tokenizer.detokenize(true_tokens)
    plotting_tools.plt_mesh(vertices_true, quads_true,
                            output_file=f'./figures/true_mesh_{i}_faces{face_count}.png')
    print(f"  True mesh: {
          quads_true.shape[1] if quads_true.dim() > 1 else 0} faces geplottet")

    start_tokens = torch.tensor(true_tokens[:8]).unsqueeze(0)

    generated_tokens = trainer.model.generate(
        point_cloud=point_cloud,
        face_count=face_count_tensor,
        start_tokens=start_tokens,
        max_length=face_count * 8 + 32
    )

    vertices_gen, quads_gen = tokenizer.detokenize(generated_tokens.tolist())
    n_gen_faces = quads_gen.shape[1] if quads_gen.dim(
    ) > 1 and quads_gen.shape[1] > 0 else 0
    plotting_tools.plt_mesh(
        vertices_gen, quads_gen, point_cloud=point_cloud[0],
        output_file=f'./figures/generated_mesh_{i}_faces{face_count}.png'
    )
    print(f"  Generated mesh: {n_gen_faces} faces (target: {face_count})")
