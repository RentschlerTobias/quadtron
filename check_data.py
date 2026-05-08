import numpy as np
import torch


def remove_duplicate_vertices(x: torch.Tensor, faces: torch.Tensor,
                              decimals: int = 8):
    """
    Remove duplicate vertices (by rounded 2D coordinate) from a mesh.

    x is the full vertex feature tensor; deduplication uses only columns 0:2.
    All feature columns are preserved in the returned tensor.

    Returns:
      new_x       — deduplicated feature tensor (same number of columns as x)
      new_faces   — face tensor with indices remapped into new_x
      n_removed   — number of vertices removed
    """
    coords_np = np.round(x[:, 0:2].numpy(), decimals=decimals)

    coord_to_id: dict = {}
    remap = np.empty(len(coords_np), dtype=int)
    keep = []

    for i, (cx, cy) in enumerate(coords_np):
        k = (cx, cy)
        if k not in coord_to_id:
            coord_to_id[k] = len(keep)
            keep.append(i)
        remap[i] = coord_to_id[k]

    new_x = x[keep]
    new_faces = torch.from_numpy(remap[faces.numpy()])
    return new_x, new_faces, len(coords_np) - len(keep)


def check_data(path: str, decimals: int = 8):
    """
    Load a list of meshes, report and remove duplicate vertices.

    Expects each mesh to have:
      .x      — vertex feature tensor, coordinates in columns 0:2
      .faces  — face index tensor of shape (4, n_faces)

    All other attributes are carried over unchanged.
    Returns the cleaned mesh list (originals are not mutated).
    """
    meshes = torch.load(path, weights_only=False)
    cleaned = []

    for i, mesh in enumerate(meshes):
        new_x, new_faces, n_removed = remove_duplicate_vertices(
            mesh.x, mesh.faces, decimals=decimals)

        if n_removed:
            print(f'[{i}] removed {n_removed} duplicate vertices '
                  f'({len(mesh.x)} → {len(new_x)})')

        m = mesh.clone()
        m.x = new_x
        m.faces = new_faces
        cleaned.append(m)

    print(f'done — {len(meshes)} meshes processed')
    return cleaned


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else './centered_blades_cleaned.pt'
    cleaned = check_data(path)
    out = path.replace('.pt', '_cleaned.pt')
    torch.save(cleaned, out)
    print(f'saved to {out}')
