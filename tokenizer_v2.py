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
            sorting_strategy: 0 = lexicographical (baseline/standard), 1 = Directed row traversal
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
                    print(f'\n adjusting max padding length to {
                          self.max_length_padding}')
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
        if self.sorting_strategy == 0:
            if self.verbose:
                print("Using lexicographical sorting (Strategy 0)")
            return self._order_quads_lexicographical(vertices, quads)
        elif self.sorting_strategy == 1:
            if self.verbose:
                print("Adjacent face based directed row ordering (Strategy 1)")
            return self._order_quads_adjacent(vertices, quads)
        else:
            raise ValueError(f"Unknown sorting_strategy={
                             self.sorting_strategy}. Must be 0 or 1.")

    def _order_quads_lexicographical(self, vertices: torch.Tensor, quads: torch.Tensor):
        """standard face sorting."""
        sorted_quads = []
        num_quads = quads.size(1)

        for i_quad in range(num_quads):
            quad = quads[:, i_quad]
            coords_quad = vertices[quad]
            xy_coords = coords_quad[:, [0, 1]]
            sort_indices = lexsort(xy_coords.T)
            lexsorted_quad = quad[sort_indices]
            lexsorted_coords = vertices[lexsorted_quad]

            ccw_ordered_quad = self._ensure_counter_clockwise(
                lexsorted_coords, lexsorted_quad)
            sorted_quads.append(ccw_ordered_quad)

        ordered_quads = torch.stack(sorted_quads)
        first_vertices = vertices[ordered_quads[:, 0]]
        quad_order = lexsort(first_vertices.T)
        return ordered_quads[quad_order]

    def _order_quads_adjacent(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Strategy 1: directed row traversal via opposite half-edge (half_edge.py)."""
        sorted_quads = order_quads_yx(vertices, quads)  # [4, n]
        reordered = self._shared_edge_vertex_order(
            vertices, sorted_quads.T)  # [n, 4]
        return reordered

    def _shared_edge_vertex_order(self, vertices: torch.Tensor, ordered_quads: torch.Tensor) -> torch.Tensor:
        """
        Reorder vertices within each face based on row direction:
          - row goes min-x → max-x: start at BL (min y, min x), CW order
            → BL, TL, TR, BR
          - row goes max-x → min-x: start at BR (min y, max x), CCW order
            → BR, TR, TL, BL

        Rows are inferred from the face sequence: faces that share 2 vertices
        with the previous face belong to the same row; otherwise a new row
        starts. For axis-aligned grids this preserves the shared-edge property
        (last 2 tokens of face N = first 2 of face N+1 reversed).
        """
        import math

        def ccw_ring(verts):
            cx = sum(vertices[v][0].item() for v in verts) / 4
            cy = sum(vertices[v][1].item() for v in verts) / 4
            return sorted(verts, key=lambda v: math.atan2(
                vertices[v][1].item() - cy, vertices[v][0].item() - cx))

        def centroid_x(verts):
            return sum(vertices[v][0].item() for v in verts) / 4

        def arrange(face_verts, direction):
            ring = ccw_ring(face_verts)
            if direction == 'cw':
                ring = ring[::-1]
                start_v = min(face_verts, key=lambda v: (
                    vertices[v][1].item(), vertices[v][0].item()))
            else:
                start_v = min(face_verts, key=lambda v: (
                    vertices[v][1].item(), -vertices[v][0].item()))
            i = ring.index(start_v)
            return ring[i:] + ring[:i]

        n = ordered_quads.shape[0]
        quads_list = [ordered_quads[i].tolist() for i in range(n)]

        rows = []
        start = 0
        for i in range(1, n):
            if len(set(quads_list[i - 1]) & set(quads_list[i])) < 2:
                rows.append((start, i))
                start = i
        rows.append((start, n))

        result = []
        for s, e in rows:
            if e - s >= 2:
                direction = 'cw' if centroid_x(quads_list[e - 1]) > centroid_x(quads_list[s]) else 'ccw'
            else:
                direction = 'cw'
            for i in range(s, e):
                result.append(arrange(quads_list[i], direction))

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

        quantized = torch.round(
            normalized * (self.quantization_levels - 1)).long()
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

    def detokenize_ordered(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like detokenize but preserves vertex order from the token sequence (no CCW reorder)."""
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
        vertices, vertex_mapping = self.unique_vertices_hash(
            all_vertices[:num_quads * 4].view(-1, 2))
        quads = torch.stack([vertex_mapping[i * 4:(i + 1) * 4] for i in range(num_quads)])
        return vertices, quads.T

    def testing(self, vertices: torch.Tensor, quads: torch.Tensor):
        tokens = self.tokenize(vertices, quads)
        recon_vertices, quads_recon = self.detokenize(tokens)

        indices_init = lexsort((vertices[:, 1], vertices[:, 0]))
        vertices_init_sorted = vertices[indices_init]
        indices_recon = lexsort((recon_vertices[:, 1], recon_vertices[:, 0]))
        recon_vertices_sorted = recon_vertices[indices_recon]

        if vertices_init_sorted.size(0) == recon_vertices_sorted.size(0):
            mse = torch.mean(
                (vertices_init_sorted - recon_vertices_sorted) ** 2)
            return mse.item() < 1e-5 and quads.size(1) == quads_recon.size(1)
        return False
