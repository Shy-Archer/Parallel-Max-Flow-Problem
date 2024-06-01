#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <climits>
#include "Graph.h"
#include "GraphGen.h"

Graph::Graph(int V)
{
    this->V = V;
    for (int i = 0; i < V; i++)
        vertex.push_back(Vertex(0, 0));
}

// Przesunięcie przepływu z przepełnionego wierzchołka u
bool Graph::push(int u)
{
    bool pushed = false;
    int n = edge.size();

    for (int i = 0; i < n; i++)
    {
        if (edge[i].u == u && edge[i].flow < edge[i].capacity && vertex[u].e_flow > 0)
        {
            if (vertex[u].height > vertex[edge[i].v].height)
            {
                int flow = min(edge[i].capacity - edge[i].flow, vertex[u].e_flow);

                vertex[u].e_flow -= flow;
                vertex[edge[i].v].e_flow += flow;
                edge[i].flow += flow;

                ReverseEdgeFlowUpdate(i, flow);

                pushed = true;
            }
        }
    }
    return pushed;
}

// Funkcja do zmiany wysokości wierzchołka u
void Graph::relabel(int u)
{
    int max_height = INT_MAX;
    int n = edge.size();

    for (int i = 0; i < n; i++)
    {
        if (edge[i].u == u && edge[i].flow < edge[i].capacity)
        {
            max_height = min(max_height, vertex[edge[i].v].height);
        }
    }

    if (max_height != INT_MAX)
    {
        vertex[u].height = max_height + 1;
    }
}

void Graph::preflow(int source)
{
    vertex[source].height = vertex.size();
    int n = edge.size();

    for (int i = 0; i < n; i++)
    {
        if (edge[i].u == source)
        {
            edge[i].flow = edge[i].capacity;

            // Inicjalizacja nadmiarowego przepływu dla sąsiedniego v
            vertex[edge[i].v].e_flow += edge[i].flow;
            edge.push_back(Edge(-edge[i].flow, 0, edge[i].v, source));
        }
    }
}

void Graph::addEdge(int u, int v, int capacity)
{
    edge.push_back(Edge(0, capacity, u, v));
}

// Aktualizacja przepływu zwrotnego dla dodanego przepływu na krawędzi i
void Graph::ReverseEdgeFlowUpdate(int i, int flow)
{
    int u = edge[i].v;
    int v = edge[i].u;
    int j = 0, n = edge.size();

    while (j < n)
    {
        if (edge[j].v == v && edge[j].u == u)
        {
            edge[j].flow -= flow;
            return;
        }
        j++;
    }

    // Dodanie krawędzi zwrotnej w grafie resztkowym
    Edge e = Edge(0, flow, u, v);
    edge.push_back(e);
}

// Funkcja do zwrócenia indeksu przepełnionego wierzchołka
int overFlowVertex(vector<Vertex>& vertex)
{
    int n = vertex.size();
    for (int i = 1; i < n - 1; i++)
        if (vertex[i].e_flow > 0)
            return i;
    return -1;
}

// Główna funkcja zwracająca maksymalny przepływ w grafie

int Graph::getMaximumFlow(int source, int sink)
{
    preflow(source);
    while (overFlowVertex(vertex) != -1)
    {
        int u = overFlowVertex(vertex);
        if (!push(u))
            relabel(u);

    }
    return vertex.back().e_flow;
}

// Funkcja do generowania losowego grafu


int main() {
 
    int V = 10;
    int numInstances = 100;
    int minV=10;
    int maxV = 100;
   

    



        vector<Graph> graphs(numInstances, Graph(V)); // Inicjalizacja każdej instancji grafu z konstruktorem parametryzowanym

        srand(time(0)); // Jednorazowe zainicjowanie generatora liczb losowych

        // Generowanie losowych grafów i przechowywanie ich w wektorze
        for (int i = 0; i < numInstances; ++i) {
            Graph& g = graphs[i];
            g.generateRandomGraph(1000, 0.25);

        }
        auto start1 = omp_get_wtime();
        // Przetwarzanie każdej instancji grafu w sekwencji
        for (int i = 0; i < numInstances; ++i) {
            Graph& g = graphs[i];

            int source = 0, sink = V - 1;

            int max_flow = g.getMaximumFlow(source, sink);
        }
        auto end1 = omp_get_wtime();
        auto duration1 = end1 - start1;
        cout << "Sequential : Execution time for instance " << ": " << duration1 << " seconds" << endl;


        auto start = omp_get_wtime();
#pragma omp parallel for 
        for (int i = 0; i < numInstances; ++i) {
            Graph& g = graphs[i];

            int source = 0, sink = V - 1;

            int max_flow = g.getMaximumFlow(source, sink);


        }
        auto end = omp_get_wtime();
        auto duration = end - start;
        cout << "OpenMp: Execution time for instance " << ": " << duration << " seconds" << endl;

       
    
    return 0;
}