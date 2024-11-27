import numpy as np
from heapq import heappush, heappop

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

def dfs(adj_list, start, visited):
    """DFS implementation to match NetworkX behavior"""
    stack = [start]
    component = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            component.append(node)
            stack.extend(sorted([n for n in adj_list[node] if n not in visited], reverse=True))
    return component

def construct_MST_edges(distance_matrix):
    """Construct MST edges using Kruskal's algorithm"""
    N = len(distance_matrix)
    edges = []
    
    # Create edge list
    for i in range(N):
        for j in range(i + 1, N):
            heappush(edges, (distance_matrix[i][j], i, j))
    
    # Build MST using Union-Find
    uf = UnionFind(N)
    mst_edges = []
    adj_list = [[] for _ in range(N)]
    
    while edges and len(mst_edges) < N - 1:
        weight, u, v = heappop(edges)
        if uf.find(u) != uf.find(v):
            mst_edges.append((u, v, weight))
            adj_list[u].append(v)
            adj_list[v].append(u)
            uf.union(u, v)
    
    return mst_edges, adj_list

def optimized_minimax_paths(distance_matrix):
    """Optimized version of Algorithm 4"""
    N = len(distance_matrix)
    result = np.zeros((N, N))
    
    # Get MST edges and adjacency list
    mst_edges, adj_list = construct_MST_edges(distance_matrix)
    
    # Sort edges by weight in descending order
    mst_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Process edges in descending order
    for u, v, weight in mst_edges:
        # Remove edge from adjacency list
        adj_list[u].remove(v)
        adj_list[v].remove(u)
        
        # Find components using DFS
        visited = set()
        tree1_nodes = dfs(adj_list, u, visited)
        tree2_nodes = dfs(adj_list, v, visited)
        
        # Update minimax values
        for p1 in tree1_nodes:
            for p2 in tree2_nodes:
                result[p1][p2] = weight
                result[p2][p1] = weight
    
    return result

def validate_results(matrix1, matrix2):
    """Compare results between two algorithms"""
    return np.allclose(matrix1, matrix2)