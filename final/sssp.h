#ifndef SSSP_H
#define SSSP_H
#include <vector>
#include <string>
struct EdgeUpdate
{
    int u, v;
    float weight;
    bool is_deletion;
};

struct Graph
{
    int n_vertices, n_edges;
    int *xadj;     // CSR row pointers
    int *adjncy;   // CSR column indices
    float *weight; // Edge weights

    Graph() : xadj(nullptr), adjncy(nullptr), weight(nullptr) {}
    ~Graph();
};

struct SSSPTree
{
    float *Dist;
    int *Parent;
    const size_t size;

    SSSPTree(size_t n);
    ~SSSPTree();
};

// Core algorithm functions
void process_batch(SSSPTree &T, const Graph &G,
                   const EdgeUpdate *updates, size_t num_updates,
                   const int *part_result, int rank);

// Utility functions
void read_graph(const std::string &filename, Graph &graph);
void partition_graph(Graph &graph, int num_parts, int *part_result);
void generate_edge_updates(std::vector<EdgeUpdate> &updates, const std::string &filename);
void distribute_updates(const std::vector<EdgeUpdate> &updates,
                        const int *part_result,
                        int rank, int n_vertices,
                        std::vector<EdgeUpdate> &local_updates);

#endif