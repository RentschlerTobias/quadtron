import numpy as np
import torch
import openmesh as om
from torch_geometric.utils import lexsort
from typing import List, Tuple, Optional
from half_edge import order_quads_yx


class Tokenizer2D:
    def __init__(self, quantization_levels, verbose=False, max_length_padding: Optional[int] = None,
                 n_start_end_tokens_repeat: Optional[int] = 8, sorting_strategy: int = 1):
        """
        Args:
            sorting_strategy: 1 = Topological DFS (baseline), 5 = Directed row traversal
        """
        self.verbose = verbose
        self.tokens_per_face = 8
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.pad_token = quantization_levels + 2
        self.vocab_size = quantization_levels + 3
        self.quantization_levels = quantization_levels
        self.n_start_end_tokens_repeat = n_start_end_tokens_repeat
        self.sorting_strategy = sorting_strategy

        self.bounds = None
        self.max_length_token_sequence = 0.0
        self.min_length_token_sequence = float("inf")

        if max_length_padding is not None:
            if max_length_padding % self.tokens_per_face != 0:
                self.max_length_padding = max_length_padding - \
                    (max_length_padding % self.tokens_per_face)
                if self.verbose:
                    print(f'\n adjusting max padding length to {self.max_length_padding}')
            else:
                self.max_length_padding = max_length_padding
        else:
            self.max_length_padding = None

    def tokenize(self, vertices: torch.Tensor, quads: torch.Tensor):
        if self.verbose:
            print('start tokenizing the mesh')

        sorted_quads = self._order_quads(vertices, quads)

        coord_sequence = self._quads_to_coords(vertices, sorted_quads)
        quantized_coords, self.bounds = self._quantize_coords(coord_sequence)
        tokens = self._build_token_sequence(quantized_coords)

        if self.max_length_token_sequence < len(tokens):
            self.max_length_token_sequence = len(tokens)
        if self.min_length_token_sequence > len(tokens):
            self.min_length_token_sequence = len(tokens)

        if self.max_length_padding is not None:
            if len(tokens) > self.max_length_padding:
                tokens = tokens[:self.max_length_padding]
            if len(tokens) < self.max_length_padding:
                fill_length = self.max_length_padding - len(tokens)
                tokens += [self.pad_token] * max(0, fill_length)

        return tokens

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor):
        if self.sorting_strategy == 1:
            if self.verbose:
                print("Using topological DFS sorting (Strategy 1)")
            return self._order_quads_topological(vertices, quads)
        elif self.sorting_strategy == 5:
            if self.verbose:
                print("Using directed row traversal (Strategy 5)")
            return self._order_quads_directed(vertices, quads)
        else:
            raise ValueError(f"Unknown sorting_strategy={self.sorting_strategy}. Must be 1 or 5.")

    def _order_quads_topological(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Strategy 1: topological DFS via OpenMesh, starting at leftmost face."""
        mesh = om.PolyMesh()
        v_np = vertices.detach().cpu().numpy()
        f_np = quads.detach().cpu().numpy().T

        vh_list = [mesh.add_vertex(np.array([v[0], v[1], 0.0])) for v in v_np]
        for f_idx in f_np:
            mesh.add_face([vh_list[i] for i in f_idx])

        n_faces = mesh.n_faces()
        visited = np.zeros(n_faces, dtype=bool)
        sorted_face_indices = []

        min_x = float('inf')
        start_fh = mesh.face_handle(0)
        for fh in mesh.faces():
            centroid_x = np.mean([mesh.point(vh)[0] for vh in mesh.fv(fh)])
            if centroid_x < min_x:
                min_x = centroid_x
                start_fh = fh

        for i in range(n_faces):
            root_fh = start_fh if i == 0 else mesh.face_handle(i)
            if not visited[root_fh.idx()]:
                stack = [root_fh]
                while stack:
                    curr_fh = stack.pop()
                    idx = curr_fh.idx()
                    if visited[idx]:
                        continue
                    visited[idx] = True
                    sorted_face_indices.append(idx)
                    for neighbor in mesh.ff(curr_fh):
                        if not visited[neighbor.idx()]:
                            stack.append(neighbor)

        topo_ordered_quads = []
        for face_idx in sorted_face_indices:
            quad = quads[:, face_idx]
            ccw_quad = self._ensure_counter_clockwise(vertices[quad], quad)
            topo_ordered_quads.append(ccw_quad)

        return torch.stack(topo_ordered_quads)

    def _order_quads_directed(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Strategy 5: directed row traversal via opposite half-edge (half_edge.py)."""
        sorted_quads = order_quads_yx(vertices, quads)  # [4, n]
        reordered = self._shared_edge_vertex_order(vertices, sorted_quads.T)  # [n, 4]
        return reordered

    def _shared_edge_vertex_order(self, vertices: torch.Tensor, ordered_quads: torch.Tensor) -> torch.Tensor:
        """
        Reorder vertices within each face so that consecutive in-row faces share
        their connecting edge as first/last tokens:
          - last  2 tokens of face N = shared edge vertices
          - first 2 tokens of face N+1 = same vertices in same order
        """
        def lex_sort(v_list):
            return sorted(v_list, key=lambda v: (vertices[v][1].item(), vertices[v][0].item()))

        n = ordered_quads.shape[0]
        result = [ordered_quads[i].tolist() for i in range(n)]

        for i in range(n):
            curr = result[i]
            curr_set = set(curr)

            entrance = None
            if i > 0:
                prev_last2 = result[i - 1][-2:]
                if all(v in curr_set for v in prev_last2):
                    entrance = list(prev_last2)

            exit_verts = None
            if i + 1 < n:
                nxt_set = set(result[i + 1])
                shared = [v for v in curr if v in nxt_set]
                if len(shared) == 2:
                    exit_verts = lex_sort(shared)

            if entrance is None and exit_verts is None:
                continue

            occupied = set(entrance or []) | set(exit_verts or [])
            remaining = lex_sort([v for v in curr if v not in occupied])

            if entrance and exit_verts:
                result[i] = entrance + exit_verts
            elif entrance:
                result[i] = entrance + remaining
            else:
                result[i] = remaining + exit_verts

        return torch.tensor(result, dtype=torch.long)

    def _ensure_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        centroid = coords.mean(dim=0)
        angles = []
        for i in range(4):
            diff = coords[i] - centroid
            angle = torch.atan2(diff[1], diff[0])
            angles.append((angle.item(), i))
        angles.sort(key=lambda x: x[0])
        ccw_order = [angles[i][1] for i in range(4)]
        return indices[ccw_order]

    def _quads_to_coords(self, vertices: torch.Tensor, ordered_quads: torch.Tensor):
        return vertices[ordered_quads].reshape(-1, 2)

    def _quantize_coords(self, coords: torch.Tensor):
        x_min, y_min = torch.min(coords, dim=0).values
        x_max, y_max = torch.max(coords, dim=0).values

        bounds = torch.tensor([x_min, y_min, x_max, y_max])
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        normalized = torch.zeros_like(coords)
        normalized[:, 0] = (coords[:, 0] - x_min) / x_range
        normalized[:, 1] = (coords[:, 1] - y_min) / y_range

        quantized = torch.round(normalized * (self.quantization_levels - 1)).long()
        quantized = torch.clamp(quantized, 0, self.quantization_levels - 1)

        return quantized, bounds

    def _build_token_sequence(self, quantized_coords: torch.Tensor) -> List[int]:
        tokens = [self.start_token] * self.n_start_end_tokens_repeat
        for coord_pair in quantized_coords:
            tokens.extend([int(coord_pair[1]), int(coord_pair[0])])
        tokens.extend([self.end_token] * self.n_start_end_tokens_repeat)
        return tokens

    def detokenize(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        coord_tokens = []
        in_coords = False
        for token in tokens:
            if token == self.start_token:
                in_coords = True
                continue
            elif token == self.end_token:
                break
            elif in_coords and token < self.quantization_levels:
                coord_tokens.append(token)

        num_full_quads = len(coord_tokens) // 8
        coord_tokens = coord_tokens[:num_full_quads * 8]
        if len(coord_tokens) == 0:
            return torch.empty(0, 2), torch.empty(4, 0)

        coord_pairs = torch.tensor(coord_tokens).reshape(-1, 2)
        x_min, y_min, x_max, y_max = self.bounds
        normalized = coord_pairs.float() / (self.quantization_levels - 1)

        all_vertices = torch.zeros_like(normalized)
        all_vertices[:, 0] = normalized[:, 1] * (x_max - x_min) + x_min
        all_vertices[:, 1] = normalized[:, 0] * (y_max - y_min) + y_min

        num_quads = len(all_vertices) // 4
        all_vertices_flat = all_vertices[:num_quads * 4].view(-1, 2)

        vertices, vertex_mapping = self.unique_vertices_hash(all_vertices_flat)

        final_quads = []
        for quad_idx in range(num_quads):
            start_idx = quad_idx * 4
            quad_vertex_indices = vertex_mapping[start_idx:start_idx + 4]
            ccw_indices = self._reorder_quad_counter_clockwise(
                vertices[quad_vertex_indices], quad_vertex_indices)
            final_quads.append(ccw_indices)

        return vertices, torch.stack(final_quads).T

    def _reorder_quad_counter_clockwise(self, coords, indices):
        centroid = coords.mean(dim=0)
        angles_and_indices = []
        for i in range(4):
            diff = coords[i] - centroid
            angle = torch.atan2(diff[1], diff[0]).item()
            angles_and_indices.append((angle, indices[i].item()))
        angles_and_indices.sort(key=lambda x: x[0])
        return torch.tensor([item[1] for item in angles_and_indices], dtype=torch.long)

    def unique_vertices_hash(self, vertices, decimals=24):
        vertices_np = vertices.numpy()
        vertices_rounded = np.round(vertices_np, decimals=decimals)
        vertex_dict = {}
        inverse_indices = []
        for i, vertex in enumerate(vertices_rounded):
            key = (vertex[0], vertex[1])
            if key not in vertex_dict:
                vertex_dict[key] = len(vertex_dict)
            inverse_indices.append(vertex_dict[key])

        unique_vertices_list = [None] * len(vertex_dict)
        for i, idx in enumerate(inverse_indices):
            if unique_vertices_list[idx] is None:
                unique_vertices_list[idx] = vertices_np[i]

        return torch.from_numpy(np.array(unique_vertices_list)).float(), \
               torch.from_numpy(np.array(inverse_indices)).long()

    def testing(self, vertices: torch.Tensor, quads: torch.Tensor):
        tokens = self.tokenize(vertices, quads)
        recon_vertices, quads_recon = self.detokenize(tokens)

        indices_init = lexsort((vertices[:, 1], vertices[:, 0]))
        vertices_init_sorted = vertices[indices_init]
        indices_recon = lexsort((recon_vertices[:, 1], recon_vertices[:, 0]))
        recon_vertices_sorted = recon_vertices[indices_recon]

        if vertices_init_sorted.size(0) == recon_vertices_sorted.size(0):
            mse = torch.mean((vertices_init_sorted - recon_vertices_sorted) ** 2)
            return mse.item() < 1e-5 and quads.size(1) == quads_recon.size(1)
        return False
