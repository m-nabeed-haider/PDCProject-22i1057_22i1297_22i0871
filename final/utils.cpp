#include "utils.h"
#include <metis.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

void read_graph(const std::string& filename, Graph& graph) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << "\n";
        exit(1);
    }

    // Read header
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    ss >> graph.n_vertices >> graph.n_edges;

    // Allocate memory
    graph.xadj = new int[graph.n_vertices + 1];
    int* adjncy = new int[2 * graph.n_edges];
    float* weight = new float[2 * graph.n_edges];

    graph.xadj[0] = 0;
    int edge_count = 0;

    for (int u = 0; u < graph.n_vertices; ++u) {
        std::getline(file, line);
        std::stringstream ls(line);

        int v;
        while (ls >> v) {
            adjncy[edge_count] = v;
            weight[edge_count] = 1.0f;
            edge_count++;
        }
        graph.xadj[u + 1] = edge_count;
    }

    graph.adjncy = adjncy;
    graph.weight = weight;
}

void generate_edge_updates(std::vector<EdgeUpdate>& updates, const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        EdgeUpdate update;
        std::stringstream ss(line);

        ss >> update.u >> update.v;
        if (ss >> update.weight) {
            update.is_deletion = false;
        }
        else {
            update.is_deletion = true;
            update.weight = 0.0f;
        }
        updates.push_back(update);
    }
}



void partition_graph(Graph& graph, int num_parts, int* part_result) {
    idx_t nv = graph.n_vertices;
    idx_t ncon = 1, objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // Convert to METIS-compatible types
    idx_t* xadj = new idx_t[graph.n_vertices + 1];
    idx_t* adjncy = new idx_t[2 * graph.n_edges];

    for (int i = 0; i < graph.n_vertices + 1; ++i)
        xadj[i] = static_cast<idx_t>(graph.xadj[i]);

    for (int i = 0; i < 2 * graph.n_edges; ++i)
        adjncy[i] = static_cast<idx_t>(graph.adjncy[i]);

    idx_t np = static_cast<idx_t>(num_parts);
    METIS_PartGraphKway(&nv, &ncon, xadj, adjncy,
        NULL, NULL, NULL, &np,
        NULL, NULL, options, &objval,
        reinterpret_cast<idx_t*>(part_result));

    delete[] xadj;
    delete[] adjncy;
}

void distribute_updates(const std::vector<EdgeUpdate>& updates,
    const int* part_result,
    int rank, int n_vertices,
    std::vector<EdgeUpdate>& local_updates) {
    local_updates.clear();
    local_updates.reserve(updates.size() / (n_vertices / 2));

    for (const auto& e : updates) {
        if ((e.u >= 0 && e.u < n_vertices && part_result[e.u] == rank) ||
            (e.v >= 0 && e.v < n_vertices && part_result[e.v] == rank)) {
            local_updates.push_back(e);
        }
    }
}