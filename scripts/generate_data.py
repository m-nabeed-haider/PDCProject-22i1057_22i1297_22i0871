import random

def generate_initial_graph():
    num_vertices = 100000
    num_edges = 150000
    edges = set()
    
    # Create spanning tree
    for i in range(1, num_vertices):
        parent = random.randint(0, i-1)
        edges.add((min(i, parent), max(i, parent)))
    
    # Add remaining edges
    while len(edges) < num_edges:
        u = random.randint(0, num_vertices-1)
        v = random.randint(0, num_vertices-1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    
    # Write to original format
    with open("initial_graph.txt", "w") as f:
        f.write(f"{num_vertices} {num_edges}\n")
        adj = {i: [] for i in range(num_vertices)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in range(num_vertices):
            f.write(f"{u}: {' '.join(map(str, adj[u]))}\n")
    
    # Write to METIS format
    with open("initial_graph_metis.txt", "w") as f:
        f.write(f"{num_vertices} {num_edges}\n")
        for u in range(num_vertices):
            f.write(" ".join(map(str, adj[u])) + "\n")

def generate_updates():
    updates = 10000
    with open("updates.txt", "w") as f:
        for _ in range(updates):
            u, v = random.sample(range(updates), 2)
            if random.random() < 0.5:  # Insert
                weight = round(random.uniform(0.1, 5.0), 1)
                f.write(f"{u} {v} {weight}\n")
            else:  # Delete
                f.write(f"{u} {v}\n")

generate_initial_graph()
generate_updates()