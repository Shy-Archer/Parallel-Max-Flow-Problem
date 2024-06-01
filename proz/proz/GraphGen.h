#pragma once


void Graph::generateRandomGraph(int maxCapacity, double density)
{
    int maxPossibleEdges = V * (V - 1); // Maksymalna liczba kraw�dzi w grafie skierowanym
    int numDesiredEdges = maxPossibleEdges * density;

    vector<vector<bool>> adjMatrix(V, vector<bool>(V, false)); // Macierz s�siedztwa do �ledzenia istniej�cych kraw�dzi

    int edgeCount = 0;
    while (edgeCount < numDesiredEdges)
    {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v && !adjMatrix[u][v]) // Zapewnienie braku p�tli i duplikat�w kraw�dzi
        {
            int capacity = rand() % maxCapacity + 1; // Pojemno�� od 1 do maxCapacity
            addEdge(u, v, capacity);
            adjMatrix[u][v] = true;
            edgeCount++;
        }
    }
}