#ifndef UTILS_H
#define UTILS_H

#include "sssp.h"
#include <string>
#include <vector>

void read_graph(const std::string& filename, Graph& graph);
void partition_graph(Graph& graph, int num_parts, int* part_result);
void generate_edge_updates(std::vector<EdgeUpdate>& updates, const std::string& filename);
void distribute_updates(const std::vector<EdgeUpdate>& updates,
                        const int* part_result,
                        int rank, int n_vertices,
                        std::vector<EdgeUpdate>& local_updates);

#endif