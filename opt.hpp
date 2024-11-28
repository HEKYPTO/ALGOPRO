#ifndef OPT_HPP
#define OPT_HPP

#include <vector>
#include <set>
#include <cstddef>

class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;

public:
    explicit UnionFind(size_t size);
    int find(int x);
    void union_sets(int x, int y);
};

std::vector<int> dfs(const std::vector<std::vector<int>>& adj_list, int start, std::set<int>& visited);

struct Edge {
    int u, v;
    double weight;
    
    Edge(int u, int v, double weight) : u(u), v(v), weight(weight) {}
    
    bool operator>(const Edge& other) const {
        return weight > other.weight;
    }
};

std::pair<std::vector<Edge>, std::vector<std::vector<int>>> 
construct_MST_edges(const std::vector<std::vector<double>>& distance_matrix);

std::vector<std::vector<double>> 
optimized_minimax_paths(const std::vector<std::vector<double>>& distance_matrix);

bool validate_results(const std::vector<std::vector<double>>& matrix1, 
                     const std::vector<std::vector<double>>& matrix2,
                     double epsilon = 1e-10);

#endif // OPT_HPP
