#pragma once


void Graph::generateRandomGraph(int maxCapacity, double density)
{
    int maxPossibleEdges = V * (V - 1); // Maksymalna liczba krawêdzi w grafie skierowanym
    int numDesiredEdges = maxPossibleEdges * density;

    vector<vector<bool>> adjMatrix(V, vector<bool>(V, false)); // Macierz s¹siedztwa do œledzenia istniej¹cych krawêdzi

    int edgeCount = 0;
    while (edgeCount < numDesiredEdges)
    {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v && !adjMatrix[u][v]) // Zapewnienie braku pêtli i duplikatów krawêdzi
        {
            int capacity = rand() % maxCapacity + 1; // Pojemnoœæ od 1 do maxCapacity
            addEdge(u, v, capacity);
            adjMatrix[u][v] = true;
            edgeCount++;
        }
    }
}