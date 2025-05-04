#ifndef SSSP_H
#define SSSP_H

#include <vector>
#include <mpi.h>
#include <omp.h>

#define INF 1e9

// SSSP Tree Data Structure
struct SSSPTree {
    std::vector<int> Parent;       // Parent vertex in the tree
    std::vector<float> Dist;       // Distance from source
    std::vector<bool> Affected;    // Affected by insertions
    std::vector<bool> AffectedDel; // Affected by deletions

    SSSPTree(int n) : Parent(n, -1), Dist(n, INF), Affected(n, false), AffectedDel(n, false) {}
};

// Edge update (insertion/deletion)
struct EdgeUpdate {
    int u, v;
    float weight;
    bool is_deletion;
};

// Graph structure (CSR format)
struct Graph {
    std::vector<int> xadj;    // CSR row pointers
    std::vector<int> adjncy;  // CSR column indices
    std::vector<float> weight; // Edge weights
    int n_vertices, n_edges;
};

// in include/sssp.h, below the Graph struct
void process_batch(SSSPTree& T,
                   const Graph& G,
                   const std::vector<EdgeUpdate>& updates,
                   const std::vector<int>& part_result,
                   int rank);


#endif
