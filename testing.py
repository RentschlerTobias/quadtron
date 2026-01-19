
import numpy as np
import matplotlib.pyplot as plt

from trainer import Trainer
import torch

# Initialisiere die Variable mit einem sehr großen Wert
# path0 = '../data/quad_data.pt'

path0 = '../data/unstructured_quad_meshes_v2.pt'
quantization = 1024
d_model = 256
n_latents = 256
batch_size = 2
num_epochs = 5
learning_rate = 1e-3
stage_layers = [2, 1, 1, 1, 1]
window_size = None
trainer = Trainer(data_path=path0, num_epochs=75, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model,
                  gradient_accumulation=4, window_size=window_size, stage_layers=stage_layers)
path = f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents{n_latents}_batch_size{batch_size}_epoch_{num_epochs}.pt'


# path = './checkpoints/q_1024_d_model_256_n_latents256_batch_size4_epoch_121.pt'
checkpoint = torch.load(path, weights_only=True)
state = checkpoint['model_state_dict']

trainer.model.load_state_dict(state)

data = next(iter(trainer.val_loader))

input_tokens = data['input_tokens'][0:1]
target_tokens = data['target_tokens'][0:1]
point_cloud = data['point_cloud'][0:1]
face_count = data['face_count'][0:1]
start = input_tokens[0:1, 0:8]


target_tokens[0, :100]
input_tokens[0, :100]

start.size()

input_tokens.size()
start.size()
point_cloud.size()
face_count.size()

generated_tokens = trainer.model.generate(
    point_cloud=point_cloud,
    face_count=face_count,
    start_tokens=start,
    max_length=2500
)
generated_tokens[:1000]

generated_tokens.size()
input_tokens.size()
info = trainer.tokenizer.info

generated_tokens.size()
vertices, quads_tensor = trainer.tokenizer.detokenize(generated_tokens, info)


# vertices, quads_tensor = trainer.tokenizer.detokenize(input_tokens[0], info)


vertices.size()


figsize = (5, 5)

plt.figure(figsize=figsize)
output_file = './generated_mesh.png'
for face in quads_tensor.T:
    coords = vertices[face].numpy()  # shape (4, 2)
    color = np.random.rand(3,)  # Random RGB color for each face
    _ = plt.fill(coords[:, 0], coords[:, 1], color=color,
                 edgecolor='gray', linewidth=0.5)
plt.axis([0, 1, 0, 1])
plt.savefig(output_file, dpi=300, transparent=True)


figsize = (5, 5)
plt.figure(figsize=figsize)

v = vertices.detach().numpy()  # [N, 2]

output_file = './generated_vertices.png'
plt.figure(figsize=(6, 6))
plt.scatter(v[:, 0], v[:, 1], s=5)
plt.axis([0, 1, 0, 1])
plt.show()
plt.savefig(output_file, dpi=300, transparent=True)
