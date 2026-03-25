/**
 * Project AETHER: Reverse Cuthill-McKee (RCM) Ordering Utility
 * * In thermochemical hypersonic flows (e.g., 11-species or 22-species air models),
 * the chemical kinetics Jacobian matrix is large and highly sparse, as species 
 * only react with specific intermediate partners. 
 * * To maximize GPU Tensor Core utilization via CUTLASS Block-Sparse GEMM, we must 
 * cluster these non-zero interactions into dense macro-blocks along the diagonal.
 * This file implements the Reverse Cuthill-McKee algorithm, a standard bandwidth 
 * reduction algorithm for symmetric sparse matrices, adapted here to generate the 
 * permutation vectors and block adjacency masks for the GPU.
 */

#ifndef PROJECT_AETHER_REVERSE_CUTHILL_MCKEE_H
#define PROJECT_AETHER_REVERSE_CUTHILL_MCKEE_H

#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

namespace aether {
namespace kinetics {

/**
 * @brief Computes the Reverse Cuthill-McKee permutation for a given adjacency list.
 * * @param adj_list The adjacency list representing chemical species interactions.
 * adj_list[i] contains the indices of species that react with species i.
 * @return std::vector<int> The RCM permutation vector mapping old indices to new indices.
 */
inline std::vector<int> compute_rcm_permutation(const std::vector<std::vector<int>>& adj_list) {
    int n = adj_list.size();
    if (n == 0) return {};

    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        degree[i] = adj_list[i].size();
    }

    std::vector<int> permutation;
    permutation.reserve(n);
    std::vector<bool> visited(n, false);

    // Lambda to sort neighbors by degree in ascending order (crucial for RCM)
    auto compare_degree = [&degree](int a, int b) {
        if (degree[a] == degree[b]) return a < b; // Tie-breaker for deterministic output
        return degree[a] < degree[b];
    };

    // Process all connected components
    for (int start_node = 0; start_node < n; ++start_node) {
        if (visited[start_node]) continue;

        // Find the node with the minimum degree in the unvisited component
        int min_degree_node = start_node;
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && degree[i] < degree[min_degree_node]) {
                min_degree_node = i;
            }
        }

        // Standard Cuthill-McKee Breadth-First Search
        std::queue<int> q;
        q.push(min_degree_node);
        visited[min_degree_node] = true;

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            permutation.push_back(current);

            // Collect unvisited neighbors
            std::vector<int> unvisited_neighbors;
            for (int neighbor : adj_list[current]) {
                if (!visited[neighbor]) {
                    unvisited_neighbors.push_back(neighbor);
                    visited[neighbor] = true; // Mark visited when adding to queue to prevent duplicates
                }
            }

            // Sort neighbors by degree in ascending order
            std::sort(unvisited_neighbors.begin(), unvisited_neighbors.end(), compare_degree);

            // Push sorted neighbors to the queue
            for (int neighbor : unvisited_neighbors) {
                q.push(neighbor);
            }
        }
    }

    // Reverse the ordering to get Reverse Cuthill-McKee
    std::reverse(permutation.begin(), permutation.end());

    return permutation;
}

/**
 * @brief Generates a block-adjacency mask for CUTLASS based on the permuted matrix.
 * * @param adj_list The original adjacency list of species interactions.
 * @param rcm_permutation The RCM permutation vector.
 * @param block_size The dimension of the square CUTLASS macro-blocks (e.g., 16, 32).
 * @return std::vector<std::vector<int>> A 2D binary mask indicating active blocks.
 */
inline std::vector<std::vector<int>> generate_block_sparse_mask(
    const std::vector<std::vector<int>>& adj_list,
    const std::vector<int>& rcm_permutation,
    int block_size
) {
    int n = adj_list.size();
    
    // Create inverse permutation mapping: new_index -> old_index
    std::vector<int> inverse_perm(n);
    for (int i = 0; i < n; ++i) {
        inverse_perm[rcm_permutation[i]] = i;
    }

    int num_blocks = (n + block_size - 1) / block_size;
    
    // Initialize empty block mask
    std::vector<std::vector<int>> block_mask(num_blocks, std::vector<int>(num_blocks, 0));

    // Populate the block mask based on the reordered adjacency list
    for (int old_i = 0; old_i < n; ++old_i) {
        int new_i = inverse_perm[old_i];
        int block_row = new_i / block_size;
        
        // A species always reacts with itself (diagonal block)
        block_mask[block_row][block_row] = 1;

        for (int old_j : adj_list[old_i]) {
            int new_j = inverse_perm[old_j];
            int block_col = new_j / block_size;
            
            // Mark the macro-block as active
            block_mask[block_row][block_col] = 1;
        }
    }

    return block_mask;
}

/**
 * @brief Helper to count the number of active blocks for GPU memory allocation.
 */
inline int count_active_blocks(const std::vector<std::vector<int>>& block_mask) {
    int count = 0;
    for (const auto& row : block_mask) {
        for (int val : row) {
            if (val > 0) count++;
        }
    }
    return count;
}

} // namespace kinetics
} // namespace aether

#endif // PROJECT_AETHER_REVERSE_CUTHILL_MCKEE_H