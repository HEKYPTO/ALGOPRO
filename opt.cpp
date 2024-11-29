#include "opt.hpp"
#include <cmath>
#include <queue>
#include <stack>
#include <algorithm>

UnionFind::UnionFind(size_t size) : parent(size), rank(size, 0) {
    for (size_t i = 0; i < size; ++i) {
        parent[i] = i;
    }
}

int UnionFind::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

void UnionFind::union_sets(int x, int y) {
    int px = find(x);
    int py = find(y);
    
    if (px == py) return;
    
    if (rank[px] < rank[py]) {
        std::swap(px, py);
    }
    parent[py] = px;
    if (rank[px] == rank[py]) {
        rank[px]++;
    }
}

std::vector<int> dfs(const std::vector<std::vector<int>>& adj_list, int start, std::set<int>& visited) {
    std::vector<int> component;
    std::stack<int> stack;
    stack.push(start);
    
    while (!stack.empty()) {
        int node = stack.top();
        stack.pop();
        
        if (visited.find(node) == visited.end()) {
            visited.insert(node);
            component.push_back(node);
            
            std::vector<int> neighbors;
            for (int n : adj_list[node]) {
                if (visited.find(n) == visited.end()) {
                    neighbors.push_back(n);
                }
            }
            std::sort(neighbors.begin(), neighbors.end(), std::greater<int>());
            for (int n : neighbors) {
                stack.push(n);
            }
        }
    }
    return component;
}

std::pair<std::vector<Edge>, std::vector<std::vector<int>>> 
construct_MST_edges(const std::vector<std::vector<double>>& distance_matrix) {
    size_t N = distance_matrix.size();
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> edges;
    
    // Create edge list
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            edges.push(Edge(i, j, distance_matrix[i][j]));
        }
    }
    
    // Build MST using Union-Find
    UnionFind uf(N);
    std::vector<Edge> mst_edges;
    std::vector<std::vector<int>> adj_list(N);
    
    while (!edges.empty() && mst_edges.size() < N - 1) {
        Edge edge = edges.top();
        edges.pop();
        
        if (uf.find(edge.u) != uf.find(edge.v)) {
            mst_edges.push_back(edge);
            adj_list[edge.u].push_back(edge.v);
            adj_list[edge.v].push_back(edge.u);
            uf.union_sets(edge.u, edge.v);
        }
    }
    
    return {mst_edges, adj_list};
}

std::vector<std::vector<double>> 
optimized_minimax_paths(const std::vector<std::vector<double>>& distance_matrix) {
    size_t N = distance_matrix.size();
    std::vector<std::vector<double>> result(N, std::vector<double>(N, 0.0));
    
    // Get MST edges and adjacency list
    auto [mst_edges, adj_list] = construct_MST_edges(distance_matrix);
    
    // Sort edges by weight in descending order
    std::sort(mst_edges.begin(), mst_edges.end(), 
              [](const Edge& a, const Edge& b) { return a.weight > b.weight; });
    
    // Process edges in descending order
    for (const Edge& edge : mst_edges) {
        int u = edge.u;
        int v = edge.v;
        double weight = edge.weight;
        
        // Remove edge from adjacency list
        adj_list[u].erase(std::remove(adj_list[u].begin(), adj_list[u].end(), v), adj_list[u].end());
        adj_list[v].erase(std::remove(adj_list[v].begin(), adj_list[v].end(), u), adj_list[v].end());
        
        // Find components using DFS
        std::set<int> visited;
        std::vector<int> tree1_nodes = dfs(adj_list, u, visited);
        std::vector<int> tree2_nodes = dfs(adj_list, v, visited);
        
        // Update minimax values
        for (int p1 : tree1_nodes) {
            for (int p2 : tree2_nodes) {
                result[p1][p2] = weight;
                result[p2][p1] = weight;
            }
        }
    }
    
    return result;
}

bool validate_results(const std::vector<std::vector<double>>& matrix1, 
                     const std::vector<std::vector<double>>& matrix2,
                     double epsilon) {
    if (matrix1.size() != matrix2.size()) return false;
    for (size_t i = 0; i < matrix1.size(); ++i) {
        if (matrix1[i].size() != matrix2[i].size()) return false;
        for (size_t j = 0; j < matrix1[i].size(); ++j) {
            if (std::abs(matrix1[i][j] - matrix2[i][j]) > epsilon) return false;
        }
    }
    return true;
}
