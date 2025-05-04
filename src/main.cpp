#include "sssp.h"
#include "utils.h"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Graph graph;
    std::vector<idx_t> part_result; 
    std::vector<EdgeUpdate> updates, local_updates;

    // Rank 0 reads and partitions the graph
    if (rank == 0) {
        read_graph("datasets/soc-Orkut.txt", graph);
        part_result.resize(graph.n_vertices);
        partition_graph(graph, num_procs, part_result);
        generate_edge_updates(updates, 1000000, graph.n_vertices); // 1M updates
    }




    // Broadcast partition result and graph metadata
    MPI_Bcast(&graph.n_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.n_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    part_result.resize(graph.n_vertices);
    MPI_Bcast(part_result.data(), graph.n_vertices, MPI_INT, 0, MPI_COMM_WORLD);



	// After broadcasting part_result and before distributing updates:

// Broadcast graph data (xadj, adjncy, weight)
	int xadj_size, adjncy_size, weight_size;
	if (rank == 0) {
		xadj_size = graph.xadj.size();
		adjncy_size = graph.adjncy.size();
		weight_size = graph.weight.size();
	}
	MPI_Bcast(&xadj_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&adjncy_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&weight_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Resize vectors on non-root ranks
	if (rank != 0) {
		graph.xadj.resize(xadj_size);
		graph.adjncy.resize(adjncy_size);
		graph.weight.resize(weight_size);
	}

	// Broadcast the actual data
	MPI_Bcast(graph.xadj.data(), xadj_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(graph.adjncy.data(), adjncy_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(graph.weight.data(), weight_size, MPI_FLOAT, 0, MPI_COMM_WORLD);


    // Distribute edge updates to relevant processes
    distribute_updates(updates, part_result, rank, local_updates);

    // Initialize SSSP tree
    SSSPTree sssp_tree(graph.n_vertices);
    if (rank == 0) sssp_tree.Dist[0] = 0; // Source vertex


	// synchronize and then print from rank 0
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		std::cout << "Final distances from source 0:\n";
		for (int i = 0; i < graph.n_vertices; ++i) {
		    float d = sssp_tree.Dist[i];
		    if (d >= INF) std::cout << i << ": INF\n";
		    else           std::cout << i << ": " << d << "\n";
		}
	}


    // Process updates in parallel
    #pragma omp parallel
    {
        #pragma omp single
        process_batch(sssp_tree, graph, local_updates, part_result, rank);
    }

    MPI_Finalize();
    return 0;
}
