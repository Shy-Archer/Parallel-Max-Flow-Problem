#pragma once
using namespace std;

struct Edge
{
    int flow;
    int capacity;
    int u, v;

    Edge(int flow, int capacity, int u, int v)
    {
        this->flow = flow;
        this->capacity = capacity;
        this->u = u;
        this->v = v;
    }
};

struct Vertex
{
    int height, e_flow;

    Vertex(int height, int e_flow)
    {
        this->height = height;
        this->e_flow = e_flow;
    }
};

// To represent a flow network
class Graph
{
    int V;
    vector<Vertex> vertex;

    bool push(int u);
    void relabel(int u);
    void preflow(int source);
    void ReverseEdgeFlowUpdate(int i, int flow);

public:
    vector<Edge> edge;
    Graph(int V);
    void addEdge(int u, int v, int w);
    int getMaximumFlow(int source, int sink);
    void generateRandomGraph(int numEdges, double density);
};