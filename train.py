
from trainer import Trainer
import torch


def train():

    # path_meshes = '../data/unstructured_quad_meshes_v2.pt'
    # path_meshes = '../data/structured_quad_meshes_pre_selected_v2.pt'
    # path_meshes = '../data/new_checkpoints/checkpoint_mesh_720.pt'

    path_meshes = './centered_blades_cleaned.pt'

    quantization = 256
    d_model = 512
    n_latents = d_model
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    stage_layers = [8, 8, 8]
    window_size = None
    gradient_accumulation = None
    n_heads = 8  # 512 / 64 = 8 channels/head (paper: 64)
    verbose = False
    sorting_strategy = 5
    max_val_samples = 3

    trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                      gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose, sorting_strategy=sorting_strategy, max_val_samples=max_val_samples)

    trainer.training()


if __name__ == "__main__":
    train()
