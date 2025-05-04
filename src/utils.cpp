#include "utils.h"
#include <fstream>
#include <metis.h>
#include <algorithm>

void read_graph(const std::string& filename, Graph& graph) {
    std::ifstream file(filename);
    file >> graph.n_vertices >> graph.n_edges;

    // Temporary adjacency lists
    std::vector<std::vector<int>> adj(graph.n_vertices);
    std::vector<std::vector<float>> weights(graph.n_vertices);

    // Read edges and mirror them
    for (int u = 0; u < graph.n_vertices; ++u) {
        int degree;
        file >> degree;
        for (int j = 0; j < degree; ++j) {
            int v;
            file >> v;
            v--; // Convert to 0-based

            // Add edge u -> v
            adj[u].push_back(v);
            weights[u].push_back(1.0f);

            // Mirror edge v -> u
            adj[v].push_back(u);
            weights[v].push_back(1.0f);
        }
    }

    // Build CSR format
    graph.xadj.resize(graph.n_vertices + 1);
    graph.xadj[0] = 0;
    for (int i = 0; i < graph.n_vertices; ++i) {
        graph.xadj[i + 1] = graph.xadj[i] + adj[i].size();
    }

    graph.adjncy.clear();
    graph.weight.clear();
    for (int i = 0; i < graph.n_vertices; ++i) {
        graph.adjncy.insert(graph.adjncy.end(), adj[i].begin(), adj[i].end());
        graph.weight.insert(graph.weight.end(), weights[i].begin(), weights[i].end());
    }

    // Update total edges (now doubled)
    graph.n_edges = graph.adjncy.size() / 2;
}

void partition_graph(Graph& graph, int num_parts, std::vector<idx_t>& part_result) {
    idx_t nvtxs = graph.n_vertices;
    idx_t ncon = 1;
    idx_t objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // Convert graph data to METIS's idx_t type
    std::vector<idx_t> xadj_idx(graph.xadj.begin(), graph.xadj.end());
    std::vector<idx_t> adjncy_idx(graph.adjncy.begin(), graph.adjncy.end());
    idx_t nparts = num_parts;

    part_result.resize(nvtxs);
    METIS_PartGraphKway(&nvtxs, &ncon, xadj_idx.data(), adjncy_idx.data(),
                        NULL, NULL, NULL, &nparts, NULL, NULL, options,
                        &objval, part_result.data());
}

void generate_edge_updates(std::vector<EdgeUpdate>& updates, int count, int n_vertices) {
    for (int i = 0; i < count; ++i) {
        EdgeUpdate e;
        e.u = rand() % n_vertices;
        e.v = rand() % n_vertices;
        e.weight = 1.0; // Example weight
        e.is_deletion = (rand() % 2 == 0);
        updates.push_back(e);
    }
}

void distribute_updates(const std::vector<EdgeUpdate>& updates, const std::vector<int>& part_result, 
                        int rank, std::vector<EdgeUpdate>& local_updates) {
    for (const auto& e : updates) {
        if (part_result[e.u] == rank || part_result[e.v] == rank) {
            local_updates.push_back(e);
        }
    }
}



