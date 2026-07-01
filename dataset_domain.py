"""
dataset_domain.py

Dataset für Domain-Partition Training (kompatibel mit tokenizer-sorting Trainer).
Ladt vorverarbeitete Daten (domain_data.pt) und tokenisiert sie.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer_domain import DomainTokenizer
from typing import Optional, List, Dict
from tqdm import tqdm


class DomainMeshData(Dataset):
    """
    Dataset für Domain-Partition Meshtron.
    API-kompatibel mit MeshData (dataset.py).
    """

    def __init__(
        self,
        meshes: List[Dict],
        tokenizer: DomainTokenizer,
        max_seq_length: Optional[int] = None,
        n_sample_points: int = 1500,
        verbose: bool = True,
    ):
        self.meshes = meshes
        self.tokenizer = tokenizer
        self.n_sample_points = n_sample_points
        self.data: List[List[int]] = []
        self.face_count: List[int] = []

        if verbose:
            print("Starte Domain-Partition Tokenisierung...")

        for i in tqdm(range(len(meshes)), desc="domain meshes", disable=not verbose):
            mesh = meshes[i]
            tokens = tokenizer.tokenize(mesh)
            self.data.append(tokens)
            self.face_count.append(mesh['faces'].shape[1])

        if max_seq_length is None:
            self.max_seq_length = tokenizer.max_length_token_sequence
        else:
            self.max_seq_length = max_seq_length

        self.min_seq_length = tokenizer.min_length_token_sequence

        if verbose:
            print(f"\nMax Sequenzlänge: {self.max_seq_length}\nMin Sequenzlänge: {self.min_seq_length}")

    def _sample_point_cloud(self, tri_coordinates: torch.Tensor) -> torch.Tensor:
        """Sample point cloud: keep boundary, fill rest with interior + noise."""
        mask = tri_coordinates[:, 2] != 2
        boundary_points = tri_coordinates[mask, :2]
        interior_points = tri_coordinates[~mask, :2]

        num_boundary = boundary_points.size(0)
        num_interior = interior_points.size(0)
        remaining = self.n_sample_points - num_boundary

        point_cloud = torch.ones([self.n_sample_points, 2]) * (-1)
        if num_boundary >= self.n_sample_points:
            idx = torch.randint(0, num_boundary, [self.n_sample_points])
            point_cloud[:, :] = boundary_points[idx, :]
        else:
            points = [boundary_points]
            if num_interior >= remaining:
                idx = torch.randint(0, num_interior, (remaining,))
                points.append(interior_points[idx, :])
            else:
                repeat_factor = (remaining + num_interior - 1) // num_interior
                interior_repeated = interior_points.repeat(repeat_factor, 1)
                noise = torch.rand_like(interior_repeated) * 0.001
                noisy_interior = interior_repeated + noise
                idx = torch.randint(0, noisy_interior.size(0), (remaining,))
                points.append(noisy_interior[idx, :])
            point_cloud = torch.cat(points, dim=0)

        # Normalisierung wie in dataset.py: [-1, 1]
        all_coords = tri_coordinates[:, :2]
        center = (all_coords.max(dim=0).values + all_coords.min(dim=0).values) / 2
        scale = (all_coords.max(dim=0).values - all_coords.min(dim=0).values).max().clamp(min=1e-6)
        point_cloud = (point_cloud - center) / scale * 2

        return point_cloud

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        mesh = self.meshes[idx]
        point_cloud = self._sample_point_cloud(mesh['tri_coordinates'])
        num_faces = self.face_count[idx]
        pad_token = self.tokenizer.pad_token

        # Padding
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        tokens_padded = tokens + [pad_token] * (self.max_seq_length - len(tokens))
        tokens_tensor = torch.LongTensor(tokens_padded)

        input_tokens = tokens_tensor[:-1]
        target_tokens = tokens_tensor[1:]
        padding_mask = (input_tokens != pad_token).float()

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'padding_mask': padding_mask,
            'pad_token': pad_token,
            'point_cloud': torch.FloatTensor(point_cloud),
            'face_count': num_faces,
            'seq_length': len(tokens) - 1,
        }


def get_domain_loaders(
    data_path: str = '/root/repos/meshtron/domain_data.pt',
    tokenizer: Optional[DomainTokenizer] = None,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    n_sample_points: int = 1500,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Erstellt Train/Val Loader für Domain-Partition."""
    meshes = torch.load(data_path, weights_only=False)
    if tokenizer is None:
        raise ValueError("Tokenizer muss übergeben werden")

    dataset = DomainMeshData(
        meshes=meshes,
        tokenizer=tokenizer,
        n_sample_points=n_sample_points,
        verbose=True,
    )

    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, dataset.max_seq_length, dataset.min_seq_length


if __name__ == '__main__':
    tok = DomainTokenizer(quantization_r=64, quantization_a=32, sorting_strategy=0, embedding_mode=0, verbose=False)
    train_loader, val_loader, max_len, min_len = get_domain_loaders(
        tokenizer=tok, batch_size=4
    )
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("input_tokens:", batch['input_tokens'].shape)
    print("point_cloud:", batch['point_cloud'].shape)
    print("face_count:", batch['face_count'])
