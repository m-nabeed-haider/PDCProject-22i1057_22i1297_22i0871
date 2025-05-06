#include "sssp.h"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <memory>
#include <cstring>
#include <limits>

struct CommBuffers {
    std::vector<int> send_verts, recv_verts;
    std::vector<float> send_wgts, recv_wgts;
    std::vector<int> sendcounts, recvcounts;
    std::vector<int> sdispl, rdispl;

    void resize(int num_procs) {
        sendcounts.resize(num_procs);
        recvcounts.resize(num_procs);
        sdispl.resize(num_procs);
        rdispl.resize(num_procs);
    }
};

namespace {
    thread_local CommBuffers comm_buf;
    thread_local std::vector<int> local_nodes;

    inline void invalidate_node(SSSPTree& T, int node) {
        if (node >= 0 && node < T.size) {
            T.Dist[node] = std::numeric_limits<float>::infinity();
            T.Parent[node] = -1;
        }
    }

    inline void safe_relax(SSSPTree& T, int u, int v, float w) {
        if (u >= 0 && u < T.size &&
            v >= 0 && v < T.size &&
            T.Dist[v] > T.Dist[u] + w)
        {
            T.Dist[v] = T.Dist[u] + w;
            T.Parent[v] = u;
        }
    }

    inline float get_edge_weight(const Graph& G, int u, int v) {
        if (u < 0 || u >= G.n_vertices)
            return std::numeric_limits<float>::infinity();

        for (int j = G.xadj[u]; j < G.xadj[u + 1]; ++j)
            if (G.adjncy[j] == v) return G.weight[j];
        return std::numeric_limits<float>::infinity();
    }
}

void process_batch(SSSPTree& T, const Graph& G,
    const EdgeUpdate* updates, size_t num_updates,
    const int* part_result, int rank)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    comm_buf.resize(num_procs);

    // Initialize local nodes safely
    if (local_nodes.empty()) {
#pragma omp critical
        {
            if (local_nodes.empty()) {
                for (int u = 0; u < G.n_vertices; ++u) {
                    if (part_result[u] == rank)
                        local_nodes.push_back(u);
                }
            }
        }
    }

    bool global_changed;
    std::vector<char> frontier(G.n_vertices, 0);

    do {
        // Phase 1: Apply updates with bounds checking
#pragma omp parallel for
        for (size_t i = 0; i < num_updates; ++i) {
            const auto& e = updates[i];
            if (e.u < 0 || e.u >= G.n_vertices ||
                e.v < 0 || e.v >= G.n_vertices) continue;

            if (e.is_deletion) {
#pragma omp critical
                {
                    if (T.Parent[e.v] == e.u) invalidate_node(T, e.v);
                    if (T.Parent[e.u] == e.v) invalidate_node(T, e.u);
                }
            }
            else {
#pragma omp critical
                {
                    safe_relax(T, e.u, e.v, e.weight);
                    safe_relax(T, e.v, e.u, e.weight);
                }
            }
        }

        // Phase 2: Local Bellman-Ford with frontier tracking
        bool local_changed;
        do {
            local_changed = false;
            std::memset(frontier.data(), 0, G.n_vertices);

#pragma omp parallel for reduction(||:local_changed)
            for (size_t i = 0; i < local_nodes.size(); ++i) {
                int u = local_nodes[i];
                if (u < 0 || u >= G.n_vertices) continue;

                for (int j = G.xadj[u]; j < G.xadj[u + 1]; ++j) {
                    int v = G.adjncy[j];
                    float w = G.weight[j];

                    if (v >= 0 && v < G.n_vertices &&
                        T.Dist[v] > T.Dist[u] + w)
                    {
#pragma omp critical
                        {
                            if (T.Dist[v] > T.Dist[u] + w) {
                                T.Dist[v] = T.Dist[u] + w;
                                T.Parent[v] = u;
                                frontier[v] = 1;
                                local_changed = true;
                            }
                        }
                    }
                }
            }
        } while (local_changed);

        // Phase 3: Optimized boundary communication
        comm_buf.send_verts.clear();
        comm_buf.send_wgts.clear();
        std::memset(comm_buf.sendcounts.data(), 0, num_procs * sizeof(int));

#pragma omp parallel for
        for (int v = 0; v < G.n_vertices; ++v) {
            if (frontier[v] && part_result[v] != rank) {
                int dest = part_result[v];
#pragma omp critical
                {
                    if (dest >= 0 && dest < num_procs) {
                        comm_buf.sendcounts[dest]++;
                        comm_buf.send_verts.push_back(v);
                        comm_buf.send_wgts.push_back(get_edge_weight(G, T.Parent[v], v));
                    }
                }
            }
        }

        MPI_Alltoall(comm_buf.sendcounts.data(), 1, MPI_INT,
            comm_buf.recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Calculate displacements safely
        comm_buf.sdispl[0] = comm_buf.rdispl[0] = 0;
        for (int i = 1; i < num_procs; ++i) {
            comm_buf.sdispl[i] = comm_buf.sdispl[i - 1] + comm_buf.sendcounts[i - 1];
            comm_buf.rdispl[i] = comm_buf.rdispl[i - 1] + comm_buf.recvcounts[i - 1];
        }

        // Resize buffers safely
        size_t total_recv = comm_buf.rdispl.back() + comm_buf.recvcounts.back();
        comm_buf.recv_verts.resize(total_recv);
        comm_buf.recv_wgts.resize(total_recv);

        // Perform safe communication
        MPI_Alltoallv(comm_buf.send_verts.data(), comm_buf.sendcounts.data(), comm_buf.sdispl.data(), MPI_INT,
            comm_buf.recv_verts.data(), comm_buf.recvcounts.data(), comm_buf.rdispl.data(), MPI_INT,
            MPI_COMM_WORLD);

        MPI_Alltoallv(comm_buf.send_wgts.data(), comm_buf.sendcounts.data(), comm_buf.sdispl.data(), MPI_FLOAT,
            comm_buf.recv_wgts.data(), comm_buf.recvcounts.data(), comm_buf.rdispl.data(), MPI_FLOAT,
            MPI_COMM_WORLD);

        // Process received updates with validation
        global_changed = false;
#pragma omp parallel for reduction(||:global_changed)
        for (size_t i = 0; i < comm_buf.recv_verts.size(); ++i) {
            int v = comm_buf.recv_verts[i];
            if (v < 0 || v >= T.size) continue;

            float w = comm_buf.recv_wgts[i];
            int parent = T.Parent[v];

            if (parent >= 0 && parent < T.size &&
                T.Dist[v] > T.Dist[parent] + w)
            {
                T.Dist[v] = T.Dist[parent] + w;
                global_changed = true;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &global_changed, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
    } while (global_changed);
}