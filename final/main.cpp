#include "sssp.h"
#include "utils.h"
#include <mpi.h>
#include <queue>
#include <iostream>
#include <limits>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Graph graph;
    int* part_result = nullptr;
    std::vector<EdgeUpdate> updates, local_updates;
    int desired_parts = num_procs;

    // Root process initialization
    if (rank == 0) {
        read_graph("initial_graph_metis.txt", graph);

        if (graph.n_vertices < num_procs) {
            std::cerr << "Reducing parts to " << graph.n_vertices << "\n";
            desired_parts = graph.n_vertices;
        }

        part_result = new int[graph.n_vertices];
        partition_graph(graph, desired_parts, part_result);
        generate_edge_updates(updates, "updates.txt");
    }

    // Broadcast graph dimensions first
    MPI_Bcast(&graph.n_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.n_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&desired_parts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate graph data on all ranks
    if (rank != 0) {
        graph.xadj = new int[graph.n_vertices + 1];
        graph.adjncy = new int[2 * graph.n_edges];
        graph.weight = new float[2 * graph.n_edges];
    }

    // Broadcast partition array
    part_result = new int[graph.n_vertices];
    MPI_Bcast(part_result, graph.n_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast CSR arrays
    MPI_Bcast(graph.xadj, graph.n_vertices + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.adjncy, 2 * graph.n_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.weight, 2 * graph.n_edges, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Initialize SSSP tree
    SSSPTree sssp_tree(graph.n_vertices);
    if (rank == 0) {
        std::fill_n(sssp_tree.Dist, graph.n_vertices,
            std::numeric_limits<float>::infinity());
        sssp_tree.Dist[0] = 0.0f;

        std::priority_queue<std::pair<float, int>> pq;
        pq.push({ 0.0f, 0 });

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > sssp_tree.Dist[u]) continue;

            for (int j = graph.xadj[u]; j < graph.xadj[u + 1]; ++j) {
                int v = graph.adjncy[j];
                float w = graph.weight[j];

                if (v >= 0 && v < graph.n_vertices &&
                    sssp_tree.Dist[v] > d + w) {
                    sssp_tree.Dist[v] = d + w;
                    sssp_tree.Parent[v] = u;
                    pq.push({ sssp_tree.Dist[v], v });
                }
            }
        }
    }

    // Broadcast SSSP tree
    MPI_Bcast(sssp_tree.Dist, graph.n_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sssp_tree.Parent, graph.n_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute updates
    distribute_updates(updates, part_result, rank, graph.n_vertices, local_updates);

    // Process updates
#pragma omp parallel
    {
#pragma omp single
        process_batch(sssp_tree, graph,
            local_updates.data(), local_updates.size(),
            part_result, rank);
    }

    // Output results
    if (rank == 0) {
        std::cout << "Final distances:\n";
        for (int i = 0; i < graph.n_vertices; ++i) {
            if (sssp_tree.Dist[i] < std::numeric_limits<float>::infinity()) {
                std::cout << i << ": " << sssp_tree.Dist[i] << "\n";
            }
            else {
                std::cout << i << ": INF\n";
            }
        }
    }

    // Cleanup
    delete[] part_result;
    MPI_Finalize();
    return 0;
}