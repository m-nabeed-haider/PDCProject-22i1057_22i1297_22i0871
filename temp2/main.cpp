#include "sssp.h"
#include "utils.h"

#include <mpi.h>
#include <omp.h>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cstring>

// Memory-efficient data structure for boundary exchanges
struct BoundaryUpdate {
    int vertex;      // Global vertex ID
    float distance;
    int parent;      // Global parent ID
};

// Memory-efficient process_batch implementation
void process_batch(SSSPTree& T,
    const Graph& G,
    const std::vector<EdgeUpdate>& updates,
    const std::vector<int>& part_result,
    int rank,
    int global_vertex_count) {

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Keep track of only affected vertices instead of checking all vertices
    std::unordered_set<int> affected_vertices;

    // Step 1: Apply updates and build initial affected set
    for (const auto& e : updates) {
        // All updates are already in local vertex space
        if (e.is_deletion) {
            // If this edge was the parent edge of v, invalidate v
            if (T.Parent[e.v] == e.u) {
                T.Dist[e.v] = INF;
                T.Parent[e.v] = -1;
                affected_vertices.insert(e.v);
            }
            if (T.Parent[e.u] == e.v) {
                T.Dist[e.u] = INF;
                T.Parent[e.u] = -1;
                affected_vertices.insert(e.u);
            }
        }
        else {
            // Apply insertion: relax the edge
            if (T.Dist[e.v] > T.Dist[e.u] + e.weight) {
                T.Dist[e.v] = T.Dist[e.u] + e.weight;
                T.Parent[e.v] = e.u;
                affected_vertices.insert(e.v);
            }
            if (T.Dist[e.u] > T.Dist[e.v] + e.weight) {
                T.Dist[e.u] = T.Dist[e.v] + e.weight;
                T.Parent[e.u] = e.v;
                affected_vertices.insert(e.u);
            }
        }
    }

    // Store boundary information for each process
    std::vector<std::vector<int>> boundary_vertices(num_procs);

    // Identify boundary vertices for each process
    for (int local_v = 0; local_v < G.n_vertices; local_v++) {
        int global_v = G.local_to_global[local_v];
        int owner = part_result[global_v];

        if (owner != rank) {
            boundary_vertices[owner].push_back(local_v);
        }
    }

    bool global_changed;
    int iteration = 0;
    const int MAX_ITERATIONS = 20; // Safety limit to prevent infinite loops

    do {
        iteration++;
        if (iteration > MAX_ITERATIONS) {
            if (rank == 0) {
                std::cerr << "Warning: Reached maximum iterations (" << MAX_ITERATIONS
                    << "), terminating update process.\n";
            }
            break;
        }

        // Step 2: Local propagation using priority queue
        std::priority_queue<
            std::pair<float, int>,
            std::vector<std::pair<float, int>>,
            std::greater<>
        > pq;

        // Initialize queue with affected vertices
        for (int v : affected_vertices) {
            pq.push({ T.Dist[v], v });
        }

        // Clear affected for this iteration
        affected_vertices.clear();

        // Process local propagation with priority queue
        std::vector<bool> processed(G.n_vertices, false);
        while (!pq.empty()) {
            auto [dist, u] = pq.top();
            pq.pop();

            if (processed[u] || dist > T.Dist[u]) continue;
            processed[u] = true;

            // Relax outgoing edges
            for (int j = G.xadj[u]; j < G.xadj[u + 1]; ++j) {
                int v = G.adjncy[j];
                float w = G.weight[j];

                if (T.Dist[v] > T.Dist[u] + w) {
                    T.Dist[v] = T.Dist[u] + w;
                    T.Parent[v] = u;
                    affected_vertices.insert(v);
                    pq.push({ T.Dist[v], v });
                }
            }
        }

        // Step 3: Boundary communication - use more efficient approach
        // Pack boundary updates per destination process
        std::vector<std::vector<BoundaryUpdate>> updates_to_send(num_procs);

        // Extract updates for boundary vertices
        for (int dest = 0; dest < num_procs; dest++) {
            if (dest == rank) continue;

            for (int local_v : boundary_vertices[dest]) {
                int global_v = G.local_to_global[local_v];
                int local_parent = T.Parent[local_v];
                int global_parent = (local_parent != -1) ? G.local_to_global[local_parent] : -1;

                updates_to_send[dest].push_back({
                    global_v,              // Global vertex ID
                    T.Dist[local_v],       // Distance
                    global_parent          // Global parent ID
                    });
            }
        }

        // Create datatype for BoundaryUpdate
        MPI_Datatype mpi_boundary_type;
        int block_lengths[3] = { 1, 1, 1 };
        MPI_Aint displacements[3] = {
            offsetof(BoundaryUpdate, vertex),
            offsetof(BoundaryUpdate, distance),
            offsetof(BoundaryUpdate, parent)
        };
        MPI_Datatype types[3] = { MPI_INT, MPI_FLOAT, MPI_INT };
        MPI_Type_create_struct(3, block_lengths, displacements, types, &mpi_boundary_type);
        MPI_Type_commit(&mpi_boundary_type);

        // Count updates per destination
        std::vector<int> send_counts(num_procs, 0);
        for (int i = 0; i < num_procs; i++) {
            send_counts[i] = updates_to_send[i].size();
        }

        // Exchange counts
        std::vector<int> recv_counts(num_procs);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
            recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Calculate displacements
        std::vector<int> send_displs(num_procs, 0), recv_displs(num_procs, 0);
        for (int i = 1; i < num_procs; i++) {
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }

        // Flatten send buffer
        int total_send = std::accumulate(send_counts.begin(), send_counts.end(), 0);
        int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

        std::vector<BoundaryUpdate> send_buffer(total_send);
        int pos = 0;
        for (int i = 0; i < num_procs; i++) {
            for (const auto& update : updates_to_send[i]) {
                send_buffer[pos++] = update;
            }
        }

        // Perform exchange
        std::vector<BoundaryUpdate> recv_buffer(total_recv);
        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), mpi_boundary_type,
            recv_buffer.data(), recv_counts.data(), recv_displs.data(), mpi_boundary_type,
            MPI_COMM_WORLD);

        // Free custom datatype
        MPI_Type_free(&mpi_boundary_type);

        // Process received updates
        global_changed = false;
        for (const auto& update : recv_buffer) {
            int global_v = update.vertex;
            int local_v = G.global_to_local[global_v];

            // Skip if vertex is not in our local subset
            if (local_v == -1) continue;

            if (T.Dist[local_v] > update.distance) {
                T.Dist[local_v] = update.distance;

                // Convert parent from global to local if it exists in our graph
                if (update.parent != -1) {
                    int local_parent = G.global_to_local[update.parent];
                    T.Parent[local_v] = local_parent;
                }
                else {
                    T.Parent[local_v] = -1;
                }

                affected_vertices.insert(local_v);
                global_changed = true;
            }
        }

        // Clean up
        send_buffer.clear();
        recv_buffer.clear();