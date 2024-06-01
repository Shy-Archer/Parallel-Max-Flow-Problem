#pragma once
Graph* generateRandomGraph(int V, int capacity, double density) {
    Graph* g = new Graph(V, V * (V - 1) * density);
    for (int i = 0; i < V * (V - 1) * density; i++) {
        int u = rand() % V;
        int v = rand() % V;
        while (u == v) {
            v = rand() % V;
        }
        int capacity_ = rand() % capacity + 1;
        g->addEdge(u, v, capacity_);
    }
    return g;
}