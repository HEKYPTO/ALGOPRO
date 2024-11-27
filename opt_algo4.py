import numpy as np
import networkx as nx
from collections import defaultdict

def cal_all_pairs_minimax_path_matrix_optimized(distance_matrix):
    N = len(distance_matrix)
    all_pairs_minimax_matrix = np.zeros((N, N))
    MST = construct_MST_from_graph_optimized(distance_matrix)
    edges_with_weights = [(u, v, d['weight']) for u, v, d in MST.edges(data=True)]
    sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
    parent = list(range(N))
    rank = [0] * N
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    components = defaultdict(set)
    for i in range(N):
        components[i].add(i)
    
    for edge in sorted_edges:
        u, v, weight = edge
        comp_u = find(u)
        comp_v = find(v)
        
        for node1 in components[comp_u]:
            for node2 in components[comp_v]:
                all_pairs_minimax_matrix[node1, node2] = weight
                all_pairs_minimax_matrix[node2, node1] = weight
        
        smaller_comp = comp_u if len(components[comp_u]) < len(components[comp_v]) else comp_v
        larger_comp = comp_v if smaller_comp == comp_u else comp_u
        components[larger_comp].update(components[smaller_comp])
        del components[smaller_comp]
        union(u, v)
    
    return all_pairs_minimax_matrix

def construct_MST_from_graph_optimized(distance_matrix):
    N = len(distance_matrix)
    G = nx.Graph()
    rows, cols = np.triu_indices(N, k=1)
    weights = distance_matrix[rows, cols]
    edges = list(zip(rows, cols, weights))
    G.add_weighted_edges_from(edges)
    return nx.minimum_spanning_tree(G)

def verify_minimax_path_matrix(original_matrix, result_matrix):
    N = len(original_matrix)
    for i in range(N):
        for j in range(i + 1, N):
            paths = nx.all_simple_paths(nx.Graph(original_matrix), i, j)
            min_max_weight = float('inf')
            for path in paths:
                path_max_weight = max(original_matrix[path[k]][path[k+1]] for k in range(len(path)-1))
                min_max_weight = min(min_max_weight, path_max_weight)
            assert abs(result_matrix[i][j] - min_max_weight) < 1e-10, f"Mismatch at ({i},{j}): got {result_matrix[i][j]}, expected {min_max_weight}"
    return True