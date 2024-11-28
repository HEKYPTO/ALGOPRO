#include <gtest/gtest.h>
#include "opt.hpp"
#include <random>
#include <chrono>

// Test small graph
TEST(OptTest, SmallGraphTest) {
    std::vector<std::vector<double>> distance_matrix = {
        {0, 1, 2},
        {1, 0, 3},
        {2, 3, 0}
    };
    
    auto result = optimized_minimax_paths(distance_matrix);
    
    // Verify matrix properties
    ASSERT_EQ(result.size(), 3);
    for (const auto& row : result) {
        ASSERT_EQ(row.size(), 3);
    }
    
    // Verify symmetry
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result.size(); ++j) {
            EXPECT_DOUBLE_EQ(result[i][j], result[j][i]);
        }
    }
    
    // Verify diagonal is zero
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i][i], 0.0);
    }
}

// Test medium graph
TEST(OptTest, MediumGraphTest) {
    std::vector<std::vector<double>> distance_matrix = {
        {0, 1, 2, 3, 4},
        {1, 0, 5, 6, 7},
        {2, 5, 0, 8, 9},
        {3, 6, 8, 0, 10},
        {4, 7, 9, 10, 0}
    };
    
    auto result = optimized_minimax_paths(distance_matrix);
    
    // Verify matrix properties
    ASSERT_EQ(result.size(), 5);
    for (const auto& row : result) {
        ASSERT_EQ(row.size(), 5);
    }
    
    // Verify symmetry
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result.size(); ++j) {
            EXPECT_DOUBLE_EQ(result[i][j], result[j][i]);
        }
    }
    
    // Verify diagonal is zero
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i][i], 0.0);
    }
}

// Test UnionFind
TEST(OptTest, UnionFindTest) {
    UnionFind uf(5);
    
    EXPECT_NE(uf.find(0), uf.find(1));
    uf.union_sets(0, 1);
    EXPECT_EQ(uf.find(0), uf.find(1));
    
    EXPECT_NE(uf.find(1), uf.find(2));
    uf.union_sets(1, 2);
    EXPECT_EQ(uf.find(0), uf.find(2));
    
    EXPECT_NE(uf.find(3), uf.find(4));
    uf.union_sets(3, 4);
    EXPECT_EQ(uf.find(3), uf.find(4));
    
    EXPECT_NE(uf.find(0), uf.find(3));
}

// Test edge cases
TEST(OptTest, EdgeCasesTest) {
    // Test 1x1 matrix
    std::vector<std::vector<double>> single_node = {{0}};
    auto single_result = optimized_minimax_paths(single_node);
    EXPECT_EQ(single_result.size(), 1);
    EXPECT_EQ(single_result[0].size(), 1);
    EXPECT_DOUBLE_EQ(single_result[0][0], 0);
    
    // Test 2x2 matrix
    std::vector<std::vector<double>> two_nodes = {
        {0, 1},
        {1, 0}
    };
    auto two_result = optimized_minimax_paths(two_nodes);
    EXPECT_EQ(two_result.size(), 2);
    for (const auto& row : two_result) {
        EXPECT_EQ(row.size(), 2);
    }
    EXPECT_DOUBLE_EQ(two_result[0][1], 1);
    EXPECT_DOUBLE_EQ(two_result[1][0], 1);
    EXPECT_DOUBLE_EQ(two_result[0][0], 0);
    EXPECT_DOUBLE_EQ(two_result[1][1], 0);
}

// Performance test
TEST(OptTest, PerformanceTest) {
    const int size = 50;  // Reduced size for quicker tests
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(1.0, 1000.0);
    
    std::vector<std::vector<double>> large_matrix(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
        large_matrix[i][i] = 0;
        for (int j = i + 1; j < size; ++j) {
            large_matrix[i][j] = dis(gen);
            large_matrix[j][i] = large_matrix[i][j];
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = optimized_minimax_paths(large_matrix);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken for " << size << "x" << size << " matrix: " 
              << duration.count() << "ms" << std::endl;
    
    // Verify result properties
    EXPECT_EQ(result.size(), size);
    for (const auto& row : result) {
        EXPECT_EQ(row.size(), size);
    }
    
    // Verify symmetry and diagonal
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(result[i][i], 0.0);
        for (int j = i + 1; j < size; ++j) {
            EXPECT_DOUBLE_EQ(result[i][j], result[j][i]);
        }
    }
}

// Test MST construction
TEST(OptTest, MSTConstructionTest) {
    std::vector<std::vector<double>> distance_matrix = {
        {0, 1, 2, 3},
        {1, 0, 4, 5},
        {2, 4, 0, 6},
        {3, 5, 6, 0}
    };
    
    auto [mst_edges, adj_list] = construct_MST_edges(distance_matrix);
    
    // MST should have n-1 edges for n vertices
    EXPECT_EQ(mst_edges.size(), distance_matrix.size() - 1);
    
    // Verify adjacency list properties
    EXPECT_EQ(adj_list.size(), distance_matrix.size());
    
    // Count total edges in adjacency list (should be 2 * (n-1) as each edge appears twice)
    int total_edges = 0;
    for (const auto& adj : adj_list) {
        total_edges += adj.size();
    }
    EXPECT_EQ(total_edges, 2 * (distance_matrix.size() - 1));
}
