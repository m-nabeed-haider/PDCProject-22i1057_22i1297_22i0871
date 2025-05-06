#include "sssp.h"
#include "utils.h"

#include <mpi.h>
#include <queue>
#include <functional>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Graph graph;
    std::vector<idx_t> part_result;
    std::vector<EdgeUpdate> updates, local_updates;

    // 1) Rank 0: read graph, choose #parts, partition, and generate updates
    int desired_parts = num_procs;
    if (rank == 0) {
        read_graph("Serial/initial_graph_metis.txt", graph);

        if (graph.n_vertices < num_procs) {
            std::cerr << "*** Warning: only "
                      << graph.n_vertices << " vertices but "
                      << num_procs << " ranks; reducing parts to "
                      << graph.n_vertices << "\n";
            desired_parts = graph.n_vertices;
        }

        part_result.resize(graph.n_vertices);
        partition_graph(graph, desired_parts, part_result);

        
        generate_edge_updates(updates,"Serial/updates.txt");
        for (auto& e : updates) {
            std::cout << e.is_deletion;
            std::cout << e.u << " ";
            std::cout << e.v << " " << std::endl;
        }
    }

    // 2) Broadcast actual #parts

    MPI_Bcast(&desired_parts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 3) Broadcast graph sizes (so each rank can resize)
    MPI_Bcast(&graph.n_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.n_edges,    1, MPI_INT, 0, MPI_COMM_WORLD);

    // 4) Build tree data structure on all ranks
    SSSPTree sssp_tree(graph.n_vertices);

    // 5) Broadcast partition array
    part_result.resize(graph.n_vertices);
    MPI_Bcast(part_result.data(),
              graph.n_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // 6) Broadcast CSR arrays
    int xadj_sz, adjncy_sz, w_sz;
    if (rank == 0) 
    {
        xadj_sz   = graph.xadj.size();
        adjncy_sz = graph.adjncy.size();
        w_sz      = graph.weight.size();
    }
    MPI_Bcast(&xadj_sz,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&adjncy_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&w_sz,      1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) 
    {
        graph.xadj.resize(xadj_sz);
        graph.adjncy.resize(adjncy_sz);
        graph.weight.resize(w_sz);
    }
    MPI_Bcast(graph.xadj.data(),   xadj_sz,   MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Bcast(graph.adjncy.data(), adjncy_sz, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Bcast(graph.weight.data(), w_sz,      MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 7) Rank 0: run full Dijkstra on the static graph
    if (rank == 0) 
    {
        sssp_tree.Dist[0] = 0.0f;
        std::priority_queue<
            std::pair<float,int>,
            std::vector<std::pair<float,int>>,
            std::greater<>
        > pq;
        pq.push({0.0f, 0});

        std::vector<bool> seen(graph.n_vertices, false);
        while (!pq.empty()) {
            auto [d,u] = pq.top(); pq.pop();
            if (seen[u]) continue;
            seen[u] = true;
            for (int j = graph.xadj[u]; j < graph.xadj[u+1]; ++j) {
                int v = graph.adjncy[j];
                float w = graph.weight[j];
                if (sssp_tree.Dist[v] > d + w) {
                    sssp_tree.Dist[v]   = d + w;
                    sssp_tree.Parent[v] = u;
                    pq.push({sssp_tree.Dist[v], v});
                }
            }
        }

        // Debug print: initial SSSP
        std::cout << "=== Initial SSSP distances from 0 ===\n";
        for (int i = 0; i < graph.n_vertices; ++i) {
            float d = sssp_tree.Dist[i];
            if (d >= INF) std::cout << i << ": INF\n";
            else          std::cout << i << ": " << d << "\n";
        }
        std::cout << "======================================\n\n";
    }

    // 8) Broadcast the initialized tree
    MPI_Bcast(sssp_tree.Dist.data(),
              graph.n_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sssp_tree.Parent.data(),
              graph.n_vertices, MPI_INT,   0, MPI_COMM_WORLD);

    // 9) Distribute (zero) updates and run the two‐phase update
    distribute_updates(updates, part_result, rank, local_updates);
    #pragma omp parallel
    {
        #pragma omp single
        process_batch(sssp_tree, graph, local_updates, part_result, rank);
    }

    // 10) Final print
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "=== Final SSSP distances from 0 ===\n";
        for (int i = 0; i < graph.n_vertices; ++i) {
            float d = sssp_tree.Dist[i];
            if (d >= INF) std::cout << i << ": INF\n";
            else          std::cout << i << ": " << d << "\n";
        }
        std::cout << "====================================\n";
    }
    //for (i = 0 ; i < graph.n)

    MPI_Finalize();
    return 0;
}

