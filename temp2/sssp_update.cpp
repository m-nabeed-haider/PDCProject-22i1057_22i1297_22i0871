#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <chrono>

#include "utils.h"

void update_graph(Graph& graph, const std::vector<EdgeUpdate>& updates) {
    std::unordered_map<int, std::unordered_map<int, float>> adj_map;

    for (int u = 0; u < graph.n_vertices; ++u)
        for (int i = graph.xadj[u]; i < graph.xadj[u + 1]; ++i)
            adj_map[u][graph.adjncy[i]] = graph.weight[i];

    for (const auto& update : updates) {
        if (update.is_deletion) {
            adj_map[update.u].erase(update.v);
            adj_map[update.v].erase(update.u);
        }
        else {
            adj_map[update.u][update.v] = update.weight;
            adj_map[update.v][update.u] = update.weight;
        }
    }

    std::vector<std::vector<int>>   new_adj(graph.n_vertices);
    std::vector<std::vector<float>> new_weights(graph.n_vertices);
    for (int u = 0; u < graph.n_vertices; ++u) {
        for (const auto& [v, w] : adj_map[u]) {
            new_adj[u].push_back(v);
            new_weights[u].push_back(w);
        }
    }

    graph.xadj.assign(graph.n_vertices + 1, 0);
    for (int i = 0; i < graph.n_vertices; ++i)
        graph.xadj[i + 1] = graph.xadj[i] + new_adj[i].size();

    graph.adjncy.clear();
    graph.weight.clear();
    graph.adjncy.reserve(graph.xadj.back());
    graph.weight.reserve(graph.xadj.back());

    for (int i = 0; i < graph.n_vertices; ++i) {
        graph.adjncy.insert(graph.adjncy.end(), new_adj[i].begin(), new_adj[i].end());
        graph.weight.insert(graph.weight.end(), new_weights[i].begin(), new_weights[i].end());
        std::vector<int>().swap(new_adj[i]);
        std::vector<float>().swap(new_weights[i]);
    }
}

void parallel_sssp_update(Graph& graph, int source, const std::vector<EdgeUpdate>& updates, std::vector<float>& dist) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    update_graph(graph, updates);
    dist.assign(graph.n_vertices, std::numeric_limits<float>::infinity());
    dist[source] = 0.0f;

    std::vector<bool> visited(graph.n_vertices, false);
    bool changed;

    do {
        changed = false;

#pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < graph.n_vertices; ++u) {
            if (visited[u] || dist[u] == std::numeric_limits<float>::infinity()) continue;

            visited[u] = true;
            for (int i = graph.xadj[u]; i < graph.xadj[u + 1]; ++i) {
                int v = graph.adjncy[i];
                float weight = graph.weight[i];

                if (dist[v] > dist[u] + weight) {
#pragma omp critical
                    {
                        if (dist[v] > dist[u] + weight) {
                            dist[v] = dist[u] + weight;
                            changed = true;
                        }
                    }
                }
            }
        }

        float local_changed = changed ? 1.0f : 0.0f;
        float global_changed;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        changed = (global_changed > 0.5f);

    } while (changed);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <update_file>\n";
        return 1;
    }

    MPI_Init(&argc, &argv);

    Graph graph;
    std::vector<EdgeUpdate> updates;
    std::vector<idx_t> part_result;

    read_graph(argv[1], graph);
    generate_edge_updates(updates, argv[2]);
    partition_graph(graph, 4, part_result);

    std::vector<EdgeUpdate> local_updates;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    distribute_updates(updates, part_result, rank, local_updates);

    std::vector<float> dist;
    auto start = std::chrono::high_resolution_clock::now();
    parallel_sssp_update(graph, 0, local_updates, dist);
    auto end = std::chrono::high_resolution_clock::now();

    if (dist.size() < 20) {
        for (size_t i = 0; i < dist.size(); ++i)
            std::cout << "Vertex " << i << ": " << dist[i] << "\n";
    }

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    MPI_Finalize();
    return 0;
}
