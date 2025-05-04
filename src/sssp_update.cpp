#include "sssp.h"
#include "utils.h"
#include <mpi.h>
#include <omp.h>
#include <unordered_map>

// Algorithm 2: Process edge changes (parallel)
#include <numeric>  // for std::accumulate

void process_batch(SSSPTree& T,
                   const Graph& G,
                   const std::vector<EdgeUpdate>& updates,
                   const std::vector<int>& part_result,
                   int rank) {
    // … your Step 1 + Step 2 (local updates + boundary_vertices) here …
    for (const EdgeUpdate& e : updates) {
        if (e.is_deletion) {
            // Mark affected vertices for deletion
            T.AffectedDel[e.u] = true;
            T.AffectedDel[e.v] = true;
        } else {
            // Relax inserted edge
            if (T.Dist[e.v] > T.Dist[e.u] + e.weight) {
                T.Dist[e.v] = T.Dist[e.u] + e.weight;
                T.Parent[e.v] = e.u;
                T.Affected[e.v] = true;
            }
        }
    }

    // ========== STEP 2: Local propagation (Bellman-Ford) ==========
    bool changed;
    do {
        changed = false;
        #pragma omp parallel for
        for (int u = 0; u < G.n_vertices; ++u) {
            // Only process vertices owned by this partition
            if (part_result[u] != rank) continue;

            for (int j = G.xadj[u]; j < G.xadj[u + 1]; ++j) {
                int v = G.adjncy[j];
                float w = G.weight[j];

                // Relax edge u -> v
                if (T.Dist[v] > T.Dist[u] + w) {
                    T.Dist[v] = T.Dist[u] + w;
                    T.Parent[v] = u;
                    T.Affected[v] = true;
                    changed = true;
                }
            }
        }
    } while (changed);

    // ========== Rest of your existing code (boundary exchange) ==========

    // Build boundary_vertices as before:
    std::vector<int> boundary_vertices;
    for (int v = 0; v < G.n_vertices; ++v) {
        if (T.Affected[v] && part_result[v] != rank)
            boundary_vertices.push_back(v);
    }

    // 3) Decide how many vertices each rank should *receive* from *you*
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> sendcounts(num_procs, 0);
    std::vector<std::vector<int>> by_dest(num_procs);
    for (int v : boundary_vertices) {
        int dest = part_result[v];
        by_dest[dest].push_back(v);
    }
    for (int r = 0; r < num_procs; ++r)
        sendcounts[r] = by_dest[r].size();

    // 4) Share sendcounts so you know how many to recv from each rank
    std::vector<int> recvcounts(num_procs);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                 recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // 5) Compute displacements and total sizes
    std::vector<int> sdispl(num_procs, 0), rdispl(num_procs, 0);
    for (int i = 1; i < num_procs; ++i) {
        sdispl[i] = sdispl[i-1] + sendcounts[i-1];
        rdispl[i] = rdispl[i-1] + recvcounts[i-1];
    }
    int total_send = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
    int total_recv = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

    // 6) Flatten your send buffer
    std::vector<int> sendbuf;
    sendbuf.reserve(total_send);
    for (int r = 0; r < num_procs; ++r) {
        sendbuf.insert(sendbuf.end(), by_dest[r].begin(), by_dest[r].end());
    }

    // 7) Allocate receive buffer
    std::vector<int> recvbuf(total_recv);

    // 8) Alltoallv exchange
    MPI_Alltoallv(sendbuf.data(),    sendcounts.data(), sdispl.data(), MPI_INT,
                  recvbuf.data(),    recvcounts.data(), rdispl.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // 9) Integrate everything you received
    for (int v : recvbuf) {
        if (T.Dist[v] > T.Dist[T.Parent[v]] + G.weight[v]) {
            T.Dist[v]      = T.Dist[T.Parent[v]] + G.weight[v];
            T.Affected[v]  = true;
        }
    }
}
