#ifndef UTILS_H
#define UTILS_H

#include "sssp.h"
#include <vector>
#include <string>
#include <metis.h>


// Read graph in METIS format
void read_graph(const std::string& filename, Graph& graph);

// Partition graph using METIS
void partition_graph(Graph& graph, int num_parts, std::vector<idx_t>& part_result);

// Generate edge updates (insertions/deletions)
void generate_edge_updates(std::vector<EdgeUpdate>& updates, int count, int n_vertices);

// Distribute updates to MPI processes
void distribute_updates(const std::vector<EdgeUpdate>& updates, const std::vector<int>& part_result, 
                        int rank, std::vector<EdgeUpdate>& local_updates);
                        

#endif
