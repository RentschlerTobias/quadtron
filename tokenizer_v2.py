import numpy as np
import torch
import openmesh as om
from torch_geometric.utils import lexsort
from typing import List, Tuple, Dict, Optional, Union
import math
import collections


class Tokenizer2D:
    def __init__(self, quantization_levels, verbose=False, max_length_padding: Optional[int] = None,
                 n_start_end_tokens_repeat: Optional[int] = 8, sorting_strategy: int = 0):
        """
        Args:
            sorting_strategy: 0 = Lexikographisch (Original), 1 = Topologisch (OpenMesh DFS)
        """
        self.verbose = verbose
        # 2D Quads (4 vertices, 2 coordinates each vertex)
        self.tokens_per_face = 8
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.pad_token = quantization_levels + 2
        self.vocab_size = quantization_levels + 3
        self.quantization_levels = quantization_levels
        self.n_start_end_tokens_repeat = n_start_end_tokens_repeat
        self.sorting_strategy = sorting_strategy

        # stores the coordinate bounds, needed to transform the coordinates back from the tokens
        self.bounds = None
        self.max_length_token_sequence = 0.0
        self.min_length_token_sequence = float("inf")

        if max_length_padding is not None:
            if max_length_padding % self.tokens_per_face != 0:
                self.max_length_padding = max_length_padding - \
                    (max_length_padding % self.tokens_per_face)
                if self.verbose:
                    print(
                        f'\n adjusting max padding length to {self.max_length_padding}')
            else:
                self.max_length_padding = max_length_padding
        else:
            self.max_length_padding = None

    def tokenize(self, vertices: torch.Tensor, quads: torch.Tensor):
        if self.verbose:
            print('start tokenizing the mesh')

        # Dispatcher für die Sortierung
        sorted_quads = self._order_quads(vertices, quads)

        # Ab hier: Dein Original-Workflow
        coord_sequence = self._quads_to_coords(vertices, sorted_quads)
        quantized_coords, self.bounds = self._quantize_coords(coord_sequence)
        tokens = self._build_token_sequence(quantized_coords)

        # Statistik & Padding (Original Logik)
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

    # --- Sortierungs-Management ---

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Wählt die Sortierstrategie basierend auf self.sorting_strategy."""
        if self.sorting_strategy == 0:
            if self.verbose:
                print("Using lexicographical sorting (Strategy 0)")
            return self._order_quads_lexicographical(vertices, quads)
        elif self.sorting_strategy == 1:
            if self.verbose:
                print("Using topological sorting (Strategy 1)")
            return self._order_quads_topological(vertices, quads)

        elif self.sorting_strategy == 2:
            if self.verbose:
                print(
                    "Using topological sorting x direction is pref for layered ordering (Strategy 2)")
            return self._order_quads_layered_x_pref(vertices, quads)

        else:
            if self.verbose:
                print("Using original input order (No sorting - Strategy 3)")
            return self._order_quads_none(vertices, quads)

    def _order_quads_lexicographical(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Deine ursprüngliche Sortier-Logik."""
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

    def _order_quads_topological(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Neue topologische Sortierung via OpenMesh."""
        # 1. Mesh für OpenMesh aufbauen
        mesh = om.PolyMesh()
        v_np = vertices.detach().cpu().numpy()
        f_np = quads.detach().cpu().numpy().T

        vh_list = [mesh.add_vertex(np.array([v[0], v[1], 0.0])) for v in v_np]
        for f_idx in f_np:
            mesh.add_face([vh_list[i] for i in f_idx])

        # 2. Start-Face finden (am weitesten links / min x)
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

        # 3. DFS durch die Nachbarschaften
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

                    # Nachbarn hinzufügen
                    for neighbor in mesh.ff(curr_fh):
                        if not visited[neighbor.idx()]:
                            stack.append(neighbor)

        # 4. Quads in der neuen Reihenfolge sammeln und CCW sicherstellen
        topo_ordered_quads = []
        for face_idx in sorted_face_indices:
            quad = quads[:, face_idx]
            # Deine Original CCW Funktion nutzen
            ccw_quad = self._ensure_counter_clockwise(vertices[quad], quad)
            topo_ordered_quads.append(ccw_quad)

        return torch.stack(topo_ordered_quads)

    def _order_quads_layered_x_pref(self, vertices: torch.Tensor, quads: torch.Tensor):
        """
        Strategie 2: Topologische Sortierung via BFS (Breitensuche).
        Erzeugt einen schichtweisen Aufbau und bevorzugt Nachbarn in X-Richtung.
        """
        # 1. Mesh für Topologie aufbauen
        mesh = om.PolyMesh()
        v_np = vertices.detach().cpu().numpy()
        f_np = quads.detach().cpu().numpy().T
        vh_list = [mesh.add_vertex(np.array([v[0], v[1], 0.0])) for v in v_np]
        for f_idx in f_np:
            mesh.add_face([vh_list[i] for i in f_idx])

        n_faces = mesh.n_faces()
        visited = np.zeros(n_faces, dtype=bool)
        sorted_face_indices = []

        # Hilfsfunktion: Centroid berechnen für Sortierung
        def get_centroid_x(fh):
            return np.mean([mesh.point(vh)[0] for vh in mesh.fv(fh)])

        # Alle Centroids vorab berechnen für Performance
        face_x_coords = np.array(
            [get_centroid_x(mesh.face_handle(i)) for i in range(n_faces)])

        # 2. Globaler Loop (um auch unzusammenhängende Mesh-Teile zu finden)
        for i in range(n_faces):
            root_fh = mesh.face_handle(i)
            if not visited[root_fh.idx()]:
                # BFS Startpunkt
                queue = collections.deque([root_fh])
                visited[root_fh.idx()] = True

                while queue:
                    curr_fh = queue.popleft()  # FIFO: Erzeugt Schichten/Wellenfront
                    sorted_face_indices.append(curr_fh.idx())

                    # Nachbarn finden, die noch nicht besucht wurden
                    neighbors = [n for n in mesh.ff(
                        curr_fh) if not visited[n.idx()]]

                    # NACHBARN SORTIEREN: Bevorzuge X-Fortschritt
                    # Wir sortieren die Nachbarn nach ihrem X-Wert.
                    # Dadurch werden "rechte" Nachbarn zuerst in die Queue gelegt
                    # und somit in der "Wellenfront" früher verarbeitet.
                    neighbors.sort(key=lambda n: face_x_coords[n.idx()])

                    for neighbor in neighbors:
                        if not visited[neighbor.idx()]:
                            visited[neighbor.idx()] = True
                            queue.append(neighbor)

        # 3. Quads in der ermittelten Reihenfolge sammeln (mit deiner CCW Logik)
        topo_ordered_quads = []
        for face_idx in sorted_face_indices:
            quad = quads[:, face_idx]
            ccw_quad = self._ensure_counter_clockwise(vertices[quad], quad)
            topo_ordered_quads.append(ccw_quad)

        return torch.stack(topo_ordered_quads)
    # --- Hilfsfunktionen (Unverändert aus deinem Original) ---

    def _ensure_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        centroid = coords.mean(dim=0)
        angles = []
        for i in range(4):
            vertex = coords[i]
            diff = vertex - centroid
            angle = torch.atan2(diff[1], diff[0])
            angles.append((angle.item(), i))
        angles.sort(key=lambda x: x[0])
        ccw_order = [angles[i][1] for i in range(4)]
        return indices[ccw_order]

    def _quads_to_coords(self, vertices: torch.Tensor, ordered_quads: torch.Tensor):
        quad_coords = vertices[ordered_quads]
        return quad_coords.reshape(-1, 2)

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
        tokens = []
        tokens.extend([self.start_token] * self.n_start_end_tokens_repeat)
        for coord_pair in quantized_coords:
            tokens.extend([int(coord_pair[1]), int(coord_pair[0])])
        tokens.extend([self.end_token] * self.n_start_end_tokens_repeat)
        return tokens

    def detokenize(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Deine Original Detokenisierung
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
            quad_vertex_indices = vertex_mapping[start_idx:start_idx+4]
            ccw_indices = self._reorder_quad_counter_clockwise(
                vertices[quad_vertex_indices], quad_vertex_indices)
            final_quads.append(ccw_indices)

        return vertices, torch.stack(final_quads).T

    def _order_quads_none(self, vertices: torch.Tensor, quads: torch.Tensor):
        """
        Strategie 3: Keine Sortierung der Faces. 
        Die Quads bleiben in der Reihenfolge, in der sie im Input-Tensor vorliegen.
        """
        ordered_quads = []
        num_quads = quads.size(1)

        for i in range(num_quads):
            quad = quads[:, i]
            # Wir behalten die Face-Reihenfolge bei,
            # stellen aber sicher, dass die 4 Punkte im Quad konsistent (CCW) sind.
            ccw_quad = self._ensure_counter_clockwise(vertices[quad], quad)
            ordered_quads.append(ccw_quad)

        return torch.stack(ordered_quads)
    # --- Restliche Hilfsmethoden (Original erhalten) ---

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

        return torch.from_numpy(np.array(unique_vertices_list)).float(), torch.from_numpy(np.array(inverse_indices)).long()

    def testing(self, vertices: torch.Tensor, quads: torch.Tensor):
        tokens = self.tokenize(vertices, quads)
        recon_vertices, quads_recon = self.detokenize(tokens)

        indices_init = lexsort((vertices[:, 1], vertices[:, 0]))
        vertices_init_sorted = vertices[indices_init]
        indices_recon = lexsort((recon_vertices[:, 1], recon_vertices[:, 0]))
        recon_vertices_sorted = recon_vertices[indices_recon]

        if vertices_init_sorted.size(0) == recon_vertices_sorted.size(0):
            mse = torch.mean((vertices_init_sorted - recon_vertices_sorted)**2)
            return mse.item() < 1e-5 and quads.size(1) == quads_recon.size(1)
        return False
