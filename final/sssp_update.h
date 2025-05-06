#ifndef SSSP_H
#define SSSP_H

#include <vector>
#include <limits>
#include <cstdint>

// Use a smaller float value for INF to avoid potential numerical issues
constexpr float INF = std::numeric_limits<float>::max() / 2.0f;

struct Graph
{
    int n_vertices;
    int n_edges;
    std::vector<int> xadj;
    std::vector<int> adjncy;
    std::vector<float> weight;
};

struct EdgeUpdate
{
    int u, v;
    float weight;
    bool is_deletion;
};

// Space-efficient SSSP tree representation
struct SSSPTree
{
    std::vector<float> Dist;
    std::vector<int> Parent;
    std::vector<bool> Affected; // Used to track affected vertices during updates

    SSSPTree(int n) : Dist(n, INF), Parent(n, -1), Affected(n, false) {}

    // Clear affected flags
    void resetAffected()
    {
        std::fill(Affected.begin(), Affected.end(), false);
    }
};

// Declaration for process_batch
void process_batch(SSSPTree &T, const Graph &G,
                   const std::vector<EdgeUpdate> &updates,
                   const std::vector<int> &part_result, int rank);

#endif // SSSP_H