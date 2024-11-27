import numpy as np
import networkx as nx
from collections import defaultdict

def cal_all_pairs_minimax_path_matrix_optimized(distance_matrix):
    N = len(distance_matrix)
    all_pairs_minimax_matrix = np.zeros((N, N))
    
    # Use Kruskal's algorithm directly to get sorted edges
    edges = []
    rows, cols = np.triu_indices(N, k=1)
    for i, j in zip(rows, cols):
        edges.append((i, j, distance_matrix[i, j]))
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
    
    # Initialize union-find
    parent = np.arange(N)
    rank = np.zeros(N, dtype=int)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            parent[px] = py
        else:
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
    
    # Track components with numpy arrays for better performance
    components = [np.array([i]) for i in range(N)]
    
    # Process edges
    for u, v, weight in sorted_edges:
        comp_u = find(u)
        comp_v = find(v)
        
        if comp_u != comp_v:
            # Update matrix using numpy broadcasting
            comp_u_nodes = components[comp_u]
            comp_v_nodes = components[comp_v]
            all_pairs_minimax_matrix[np.ix_(comp_u_nodes, comp_v_nodes)] = weight
            all_pairs_minimax_matrix[np.ix_(comp_v_nodes, comp_u_nodes)] = weight
            
            # Merge components more efficiently
            if len(components[comp_u]) < len(components[comp_v]):
                comp_u, comp_v = comp_v, comp_u
            components[comp_u] = np.concatenate([components[comp_u], components[comp_v]])
            components[comp_v] = components[comp_u]  # Share reference for future merges
            union(u, v)
    
    return all_pairs_minimax_matrix

def verify_minimax_path_matrix(original_matrix, result_matrix):
    N = len(original_matrix)
    G = nx.from_numpy_array(original_matrix)
    
    for i in range(N):
        for j in range(i + 1, N):
            paths = list(nx.all_simple_paths(G, i, j))
            min_max_weight = float('inf')
            for path in paths:
                path_weights = [original_matrix[path[k]][path[k+1]] for k in range(len(path)-1)]
                min_max_weight = min(min_max_weight, max(path_weights))
            
            if not np.isclose(result_matrix[i, j], min_max_weight):
                return False
            if not np.isclose(result_matrix[j, i], min_max_weight):
                return False
    return True