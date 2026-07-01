"""
reconstruct_domain.py

Rekonstruktion einer Domain-Partition aus detokenisierten Daten.
Schritte:
  1. Polar -> Cartesian fuer alle Vertex-Plaetze
  2. Merge duplicate vertices (Proximity-Clustering)
  3. Baue Faces mit unique Vertex-Indices
  4. Generiere Hermite-Spline Punkte pro Kante
  5. Erstelle blocked_mesh Data-Objekt
  6. Rufe Transfinite_Interpolation auf
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, '/root/repos/domain_partition/tools')
from transfinite_interpolation import Transfinite_Interpolation


def polar_to_cartesian(places: List[Dict], center: np.ndarray) -> np.ndarray:
    """Konvertiere polar places zu cartesian [n_places, 2]."""
    pts = []
    for p in places:
        r = p['r']
        theta = p['theta']
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def merge_duplicate_vertices(cartesian: np.ndarray, threshold: float = 1e-3) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Merge nahe beieinander liegende Vertex-Plaetze.
    
    Returns:
        unique_verts: [n_unique, 2]
        place_to_unique: dict {place_idx -> unique_idx}
    """
    n = cartesian.shape[0]
    unique_verts = []
    place_to_unique = {}
    
    for i in range(n):
        found = False
        for j, uv in enumerate(unique_verts):
            if np.linalg.norm(cartesian[i] - uv) < threshold:
                place_to_unique[i] = j
                found = True
                break
        if not found:
            place_to_unique[i] = len(unique_verts)
            unique_verts.append(cartesian[i].copy())
    
    return np.array(unique_verts, dtype=np.float32), place_to_unique


def hermite_spline_2d(P0: np.ndarray, P1: np.ndarray,
                      T0: np.ndarray, T1: np.ndarray,
                      n_points: int = 50) -> np.ndarray:
    """
    Hermite-Spline in 2D.
    P0, P1: Start/End Punkt [2]
    T0, T1: Tangentenvektoren [2]
    Returns: [n_points, 2]
    """
    t = np.linspace(0, 1, n_points)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    points = (h00[:, None] * P0 +
              h10[:, None] * T0 +
              h01[:, None] * P1 +
              h11[:, None] * T1)
    return points.astype(np.float32)


def build_hermite_edges(places: List[Dict], cartesian: np.ndarray,
                        faces_as_places: List[List[int]],
                        place_to_unique: Dict[int, int]) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Baue unique edges und Hermite-Spline Punkte.
    
    Returns:
        edge_index: [2, n_unique_edges]
        edge_points: Liste von [n_points, 2] Arrays
    """
    # Sammle alle directed edges aus den Faces
    directed_edges = []
    directed_edge_info = []  # (place0, place1, alpha_out, alpha_in, t_norm)
    
    for face in faces_as_places:
        for i in range(4):
            p0 = face[i]
            p1 = face[(i + 1) % 4]
            
            alpha_out = places[p0]['alpha_out']
            alpha_in = places[p1]['alpha_in']
            t_norm = places[p0]['t_norm']
            
            u = place_to_unique[p0]
            v = place_to_unique[p1]
            
            directed_edges.append((u, v))
            directed_edge_info.append({
                'P0': cartesian[p0],
                'P1': cartesian[p1],
                'alpha_out': alpha_out,
                'alpha_in': alpha_in,
                't_norm': t_norm,
            })
    
    # Merge undirected edges: (u,v) und (v,u) sind dieselbe physikalische Kante
    # Wir behalten die erste Richtung bei und mitteln die Tangenten
    unique_edges = {}
    
    for (u, v), info in zip(directed_edges, directed_edge_info):
        key = tuple(sorted([u, v]))
        if key not in unique_edges:
            unique_edges[key] = {
                'u': u, 'v': v,
                'P0': info['P0'],
                'P1': info['P1'],
                'alpha_out': info['alpha_out'],
                'alpha_in': info['alpha_in'],
                't_norm': info['t_norm'],
                'count': 1,
            }
        else:
            # Mitteln der Tangenten (einfacher Durchschnitt der Winkel)
            existing = unique_edges[key]
            existing['alpha_out'] = (existing['alpha_out'] + info['alpha_out']) / 2
            existing['alpha_in'] = (existing['alpha_in'] + info['alpha_in']) / 2
            existing['t_norm'] = (existing['t_norm'] + info['t_norm']) / 2
            existing['count'] += 1
    
    n_edges = len(unique_edges)
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    edge_points = []
    
    for i, (key, info) in enumerate(sorted(unique_edges.items())):
        u, v = info['u'], info['v']
        edge_index[0, i] = u
        edge_index[1, i] = v
        
        P0 = info['P0']
        P1 = info['P1']
        
        # Tangente am Start (von u weg in Richtung v)
        T0 = info['t_norm'] * np.array([np.cos(info['alpha_out']), np.sin(info['alpha_out'])])
        # Tangente am Ende (in v hinein, von u kommend)
        T1 = info['t_norm'] * np.array([np.cos(info['alpha_in']), np.sin(info['alpha_in'])])
        
        pts = hermite_spline_2d(P0, P1, T0, T1, n_points=50)
        edge_points.append(pts)
    
    return edge_index, edge_points


def reconstruct_blocked_mesh(tokenizer_output: Dict, center: np.ndarray) -> Data:
    """
    Erstelle blocked_mesh Data-Objekt aus detokenisierten Daten.
    
    Args:
        tokenizer_output: dict mit 'vertex_places', 'faces_as_places', 'n_places'
        center: [2] array (Mesh-Zentrum)
    
    Returns:
        blocked_mesh: Data mit x, faces, edge_subdomain_index, edge_subdomain_points
    """
    places = tokenizer_output['vertex_places']
    faces_as_places = tokenizer_output['faces_as_places']
    
    # 1. Polar -> Cartesian
    cartesian = polar_to_cartesian(places, center)
    
    # 2. Merge duplicates
    unique_verts, place_to_unique = merge_duplicate_vertices(cartesian, threshold=1e-3)
    
    # 3. Baue Faces mit unique indices
    n_faces = len(faces_as_places)
    faces = torch.zeros((4, n_faces), dtype=torch.long)
    for fi, face in enumerate(faces_as_places):
        for i, p in enumerate(face):
            faces[i, fi] = place_to_unique[p]
    
    # 4. Hermite-Splines fuer Kanten
    edge_index, edge_points = build_hermite_edges(places, cartesian, faces_as_places, place_to_unique)
    
    # 5. Erstelle Data-Objekt
    # x muss [n_verts, 3] sein? Transfinite_Interpolation nimmt pt[0], pt[1], 0
    # Also [n_verts, 2] reicht, weil es nur pt[0], pt[1] liest
    x = torch.from_numpy(unique_verts).float()
    
    blocked_mesh = Data(x=x, faces=faces,
                        edge_subdomain_index=edge_index,
                        edge_subdomain_points=edge_points)
    
    return blocked_mesh


def reconstruct_domain(tokenizer_output: Dict, center: np.ndarray,
                         transfinite_divisions: int = 10) -> Data:
    """
    Vollstaendige Rekonstruktion: blocked_mesh -> transfinite interpolation.
    
    Returns:
        quad_mesh: Data mit x, faces, edge_index
    """
    blocked_mesh = reconstruct_blocked_mesh(tokenizer_output, center)
    
    # Transfinite Interpolation
    # Hinweis: Transfinite_Interpolation.__init__ ruft bereits generate() und get_mesh() auf.
    # Das Ergebnis ist in interpolator.quad_mesh gespeichert.
    interpolator = Transfinite_Interpolation(blocked_mesh, mesh_size=0.4)
    quad_mesh = interpolator.quad_mesh
    
    return quad_mesh


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/root/repos/meshtron')
    from tokenizer_domain import DomainTokenizer
    
    # Test round-trip
    data = torch.load('/root/repos/meshtron/domain_data.pt', weights_only=False)
    mesh = data[0]
    
    tok = DomainTokenizer(quantization_r=64, quantization_a=32,
                           sorting_strategy=0, embedding_mode=0, verbose=False)
    tokens = tok.tokenize(mesh)
    output = tok.detokenize(tokens)
    
    center = mesh['center'].numpy()
    print(f"Center: {center}")
    print(f"Places: {output['n_places']}")
    print(f"Faces: {len(output['faces_as_places'])}")
    
    blocked = reconstruct_blocked_mesh(output, center)
    print(f"Blocked mesh: n_verts={blocked.x.shape[0]}, n_faces={blocked.faces.shape[1]}, n_edges={blocked.edge_subdomain_index.shape[1]}")
    
    # Transfinite interpolation
    try:
        quad = reconstruct_domain(output, center, transfinite_divisions=5)
        print(f"Quad mesh: n_verts={quad.x.shape[0]}, n_faces={quad.faces.shape[1]}")
    except Exception as e:
        print(f"Transfinite interpolation failed: {e}")
        import traceback
        traceback.print_exc()
