#include "sssp.h"
#include "utils.h"

#include <mpi.h>
#include <omp.h>
#include <numeric>
#include <vector>

void process_batch(SSSPTree& T,
                   const Graph& G,
                   const std::vector<EdgeUpdate>& updates,
                   const std::vector<int>& part_result,
                   int rank) {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    bool global_changed;
    do {
        // 1) Apply local insert/deletions
        for (auto& e : updates) {
            if (e.is_deletion) {
                T.AffectedDel[e.u] = T.AffectedDel[e.v] = true;
            } else {
                if (T.Dist[e.v] > T.Dist[e.u] + e.weight) {
                    T.Dist[e.v]    = T.Dist[e.u] + e.weight;
                    T.Parent[e.v]  = e.u;
                    T.Affected[e.v] = true;
                }
            }
        }

        // 2a) Local Bellman–Ford until stable
        bool local_changed;
        do {
            local_changed = false;
            #pragma omp parallel for reduction(||:local_changed)
            for (int u = 0; u < G.n_vertices; ++u) {
                if (part_result[u] != rank) continue;
                for (int j = G.xadj[u]; j < G.xadj[u+1]; ++j) {
                    int v = G.adjncy[j];
                    float w = G.weight[j];
                    if (T.Dist[v] > T.Dist[u] + w) {
                        T.Dist[v]    = T.Dist[u] + w;
                        T.Parent[v]  = u;
                        T.Affected[v] = true;
                        local_changed = true;
                    }
                }
            }
        } while (local_changed);

        // 2b) Exchange boundary relaxations
        std::vector<int> boundary;
        for (int v = 0; v < G.n_vertices; ++v) {
            if (T.Affected[v] && part_result[v] != rank)
                boundary.push_back(v);
            T.Affected[v] = false;
        }

        // group by destination
        std::vector<std::vector<int>> by_dest(num_procs);
        for (int v : boundary)
            by_dest[ part_result[v] ].push_back(v);

        // counts & alltoall
        std::vector<int> sendcounts(num_procs), recvcounts(num_procs);
        for (int r = 0; r < num_procs; ++r)
            sendcounts[r] = by_dest[r].size();
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                     recvcounts.data(), 1, MPI_INT,
                     MPI_COMM_WORLD);

        // displacements & totals
        std::vector<int> sdispl(num_procs), rdispl(num_procs);
        sdispl[0] = rdispl[0] = 0;
        for (int i = 1; i < num_procs; ++i) {
            sdispl[i] = sdispl[i-1] + sendcounts[i-1];
            rdispl[i] = rdispl[i-1] + recvcounts[i-1];
        }
        int total_send = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
        int total_recv = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

        // flatten two send buffers: verts + weights
        std::vector<int>   send_verts; send_verts.reserve(total_send);
        std::vector<float> send_wgts;  send_wgts .reserve(total_send);
        for (int r = 0; r < num_procs; ++r) {
            for (int v : by_dest[r]) {
                send_verts.push_back(v);
                int u = T.Parent[v];
                float w = INF;
                for (int j = G.xadj[u]; j < G.xadj[u+1]; ++j) {
                    if (G.adjncy[j] == v) { w = G.weight[j]; break; }
                }
                send_wgts.push_back(w);
            }
        }

        std::vector<int>   recv_verts(total_recv);
        std::vector<float> recv_wgts (total_recv);
        MPI_Alltoallv(send_verts.data(), sendcounts.data(), sdispl.data(), MPI_INT,
                      recv_verts.data(), recvcounts.data(), rdispl.data(), MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(send_wgts.data(),  sendcounts.data(), sdispl.data(), MPI_FLOAT,
                      recv_wgts.data(),  recvcounts.data(), rdispl.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);

        // integrate & check global change
        global_changed = false;
        for (int i = 0; i < total_recv; ++i) {
            int   v = recv_verts[i];
            float w = recv_wgts [i];
            int   u = T.Parent[v];
            if (T.Dist[v] > T.Dist[u] + w) {
                T.Dist[v]     = T.Dist[u] + w;
                T.Affected[v] = true;
                global_changed = true;
            }
        }
        bool any_changed;
        MPI_Allreduce(&global_changed, &any_changed, 1, MPI_CXX_BOOL,
                      MPI_LOR, MPI_COMM_WORLD);
        global_changed = any_changed;

    } while (global_changed);
}

