#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <limits>
#include <string>

using namespace std;

struct EdgeUpdate {
    int u, v;
    float weight;
    bool is_deletion;
};

using Graph = unordered_map<int, unordered_map<int, double>>;
using DistMap = unordered_map<int, double>;
using ParentMap = unordered_map<int, int>;

// Dijkstra's algorithm for initial SSSP calculation
void dijkstra_initial(const Graph& graph, int source, DistMap& dist, ParentMap& parent) {
    auto cmp = [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first;
        };
    priority_queue<pair<double, int>, vector<pair<double, int>>, decltype(cmp)> pq(cmp);

    // Initialize distances and parents
    for (const auto& [node, _] : graph) {
        dist[node] = numeric_limits<double>::infinity();
        parent[node] = -1;
    }
    dist[source] = 0.0;
    pq.emplace(0.0, source);

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top();
        pq.pop();

        if (current_dist > dist[u]) continue;

        for (const auto& [v, weight] : graph.at(u)) {
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }
}

// Update graph structure
void update_graph(Graph& graph, const string& op, int u, int v, double weight = 1.0) {
    if (op == "insert") {
        graph[u][v] = weight;
        graph[v][u] = weight;
    }
    else if (op == "delete") {
        graph[u].erase(v);
        graph[v].erase(u);
    }
}

// Update SSSP tree after a change
void update_sssp(Graph& graph, DistMap& dist, ParentMap& parent,
    const string& op, int u, int v, double weight = 1.0) {
    auto cmp = [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first;
        };
    priority_queue<pair<double, int>, vector<pair<double, int>>, decltype(cmp)> pq(cmp);

    update_graph(graph, op, u, v, weight);

    if (op == "delete") {
        if (parent[v] == u || parent[u] == v) {
            int x = dist[u] > dist[v] ? u : v;
            dist[x] = numeric_limits<double>::infinity();
            parent[x] = -1;
            pq.emplace(dist[x], x);
        }
    }
    else {
        if (dist[u] + weight < dist[v]) {
            dist[v] = dist[u] + weight;
            parent[v] = u;
            pq.emplace(dist[v], v);
        }
        if (dist[v] + weight < dist[u]) {
            dist[u] = dist[v] + weight;
            parent[u] = v;
            pq.emplace(dist[u], u);
        }
    }

    while (!pq.empty()) {
        auto [_, z] = pq.top();
        pq.pop();

        for (const auto& [neighbor, w] : graph[z]) {
            double new_dist = dist[z] + w;
            if (new_dist < dist[neighbor]) {
                dist[neighbor] = new_dist;
                parent[neighbor] = z;
                pq.emplace(new_dist, neighbor);
            }
        }
    }
}

// Read METIS format graph
void read_graph(const string& filename, Graph& graph) {
    ifstream file(filename);
    int num_vertices, num_edges;
    file >> num_vertices >> num_edges;

    string line;
    getline(file, line); // Consume remaining newline

    for (int u = 0; u < num_vertices; ++u) {
        getline(file, line);
        istringstream iss(line);
        int v;
        while (iss >> v) {
            graph[u][v] = 1.0;  // Initial weight = 1.0
            graph[v][u] = 1.0;
        }
    }
}

// Read updates from file
void generate_edge_updates(vector<EdgeUpdate>& updates, const string& filename) {
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        EdgeUpdate update;
        iss >> update.u >> update.v;

        // Check if line contains weight (insert operation)
        if (iss >> update.weight) {
            update.is_deletion = false;
        }
        else {
            update.is_deletion = true;
            update.weight = 0.0f;
        }

        updates.push_back(update);
    }
}

// Process all updates
void process_updates(Graph& graph, DistMap& dist, ParentMap& parent,
    const vector<EdgeUpdate>& updates) {
    for (const auto& update : updates) {
        if (update.is_deletion) {
            update_sssp(graph, dist, parent, "delete", update.u, update.v);
        }
        else {
            update_sssp(graph, dist, parent, "insert", update.u, update.v, update.weight);
        }
    }
}

int main() {
    Graph graph;
    DistMap dist;
    ParentMap parent;
    vector<EdgeUpdate> updates;
    const int source = 0;

    // Read initial graph
    read_graph("initial_graph_metis.txt", graph);

    // Read updates
    generate_edge_updates(updates, "updates.txt");

    // Initial SSSP calculation
    dijkstra_initial(graph, source, dist, parent);

    // Process updates
    process_updates(graph, dist, parent, updates);

    // Output final distances
    ofstream out("distances.txt");
    for (const auto& [node, d] : dist) {
        out << node << " " << d << "\n";
    }

    return 0;
}