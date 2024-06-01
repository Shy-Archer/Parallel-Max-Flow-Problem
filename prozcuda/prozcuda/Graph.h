#pragma once
struct Edge {
    int flow, capacity;
    int u, v;
    __host__ __device__ Edge(int flow = 0, int capacity = 0, int u = 0, int v = 0)
        : flow(flow), capacity(capacity), u(u), v(v) {}
};

struct Vertex {
    int h, e_flow;
    __host__ __device__ Vertex(int h = 0, int e_flow = 0) : h(h), e_flow(e_flow) {}
};

class Graph {
public:
    int v, edgeCount;
    Vertex* ver;
    Edge* edge;

    __host__ __device__ Graph(int v = 0, int edgeCount = 0);
    __host__ __device__ void addEdge(int u, int v, int capacity);
    __device__ void preflow(int s);
    __device__ bool push(int u);
    __device__ void relabel(int u);
    __device__ void updateReverseEdgeFlow(int i, int flow);
    __device__ int getMaxFlow(int s, int t);
};