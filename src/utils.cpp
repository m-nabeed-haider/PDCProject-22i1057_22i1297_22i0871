#include "utils.h"
#include <fstream>
#include <metis.h>
#include <sstream>
#include <iostream>
#include<ctime>
void read_graph(const std::string& filename, Graph& graph) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: cannot open " << filename << "\n";
        exit(1);
    }

    // Read header
    std::string header;
    std::getline(file, header);
    {
        std::stringstream hs(header);
        hs >> graph.n_vertices >> graph.n_edges;
    }

    // Temporary adjacency
    std::vector<std::vector<int>>   adj(graph.n_vertices);
    std::vector<std::vector<float>> weights(graph.n_vertices);

    for (int u = 0; u < graph.n_vertices; ++u) {
        std::string line;
        std::getline(file, line);
        if (line.empty()) { --u; continue; }

        auto cpos = line.find('#');
        if (cpos != std::string::npos)
            line.resize(cpos);
        std::stringstream ss(line);

        int deg; ss >> deg;
        for (int j = 0; j < deg; ++j) {
            int v; ss >> v;  // assume 0‑based
            adj[u].push_back(v);
            weights[u].push_back(1.0f);
            adj[v].push_back(u);
            weights[v].push_back(1.0f);
        }
    }

    // Build CSR
    graph.xadj.resize(graph.n_vertices + 1);
    graph.xadj[0] = 0;
    for (int i = 0; i < graph.n_vertices; ++i)
        graph.xadj[i+1] = graph.xadj[i] + adj[i].size();

    graph.adjncy.reserve(graph.xadj.back());
    graph.weight.reserve(graph.xadj.back());
    for (int i = 0; i < graph.n_vertices; ++i) {
        graph.adjncy.insert(graph.adjncy.end(),
                            adj[i].begin(), adj[i].end());
        graph.weight.insert(graph.weight.end(),
                            weights[i].begin(), weights[i].end());
    }

    graph.n_edges = graph.adjncy.size()/2;
}

void partition_graph(Graph& graph, int num_parts, std::vector<idx_t>& part_result) {
    idx_t nv = graph.n_vertices;
    idx_t ncon = 1, objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    std::vector<idx_t> xadj(graph.xadj.begin(), graph.xadj.end());
    std::vector<idx_t> adjncy(graph.adjncy.begin(), graph.adjncy.end());
    idx_t np = num_parts;
    part_result.resize(nv);

    METIS_PartGraphKway(&nv, &ncon,
                        xadj.data(), adjncy.data(),
                        NULL, NULL, NULL,
                        &np, NULL, NULL,
                        options,
                        &objval,
                        part_result.data());
}

void generate_edge_updates(std::vector<EdgeUpdate>& updates, int count, int n_vertices) {
    srand(time(0));
    for (int i = 0; i < count; ++i) {
        EdgeUpdate e;
        e.u = rand() % n_vertices;
        e.v = rand() % n_vertices;
        e.weight     = (rand() % n_vertices + 1) * 0.1;
        e.is_deletion = (rand()%2 == 0);
        updates.push_back(e);
    }
}

void distribute_updates(const std::vector<EdgeUpdate>& updates,
                        const std::vector<int>& part_result,
                        int rank,
                        std::vector<EdgeUpdate>& local_updates) {
    for (auto &e : updates) {
        if (part_result[e.u]==rank || part_result[e.v]==rank)
            local_updates.push_back(e);
    }
}

