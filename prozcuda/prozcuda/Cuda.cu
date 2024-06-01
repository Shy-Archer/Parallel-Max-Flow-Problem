#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>
#include "Graph.h"
#include "GraphGen.h"
//#define V 50
#define THREADS_PER_BLOCK 2560


__host__ __device__ Graph::Graph(int v, int edgeCount) {
    this->v = v;
    this->edgeCount = 0;
    ver = new Vertex[v];
    edge = new Edge[2 * edgeCount];
}

__host__ __device__ void Graph::addEdge(int u, int v, int capacity) {
    edge[edgeCount++] = Edge(0, capacity, u, v);
}

__device__ void Graph::preflow(int s) {
    ver[s].h = v;
    for (int i = 0; i < edgeCount; i++) {
        if (edge[i].u == s) {
            edge[i].flow = edge[i].capacity;
            ver[edge[i].v].e_flow += edge[i].flow;
            edge[edgeCount++] = Edge(-edge[i].flow, 0, edge[i].v, s);
        }
    }
}

__device__ int overFlowVertex(Vertex* ver, int v) {
    for (int i = 1; i < v - 1; i++) {
        if (ver[i].e_flow > 0)
            return i;
    }
    return -1;
}

__device__ void Graph::updateReverseEdgeFlow(int i, int flow) {
    int u = edge[i].v, v = edge[i].u;
    for (int j = 0; j < edgeCount; j++) {
        if (edge[j].v == v && edge[j].u == u) {
            edge[j].flow -= flow;
            return;
        }
    }
    edge[edgeCount++] = Edge(0, flow, u, v);
}

__device__ bool Graph::push(int u) {
    for (int i = 0; i < edgeCount; i++) {
        if (edge[i].u == u) {
            if (edge[i].flow == edge[i].capacity)
                continue;
            if (ver[u].h > ver[edge[i].v].h) {
                int flow = min(edge[i].capacity - edge[i].flow, ver[u].e_flow);
                ver[u].e_flow -= flow;
                ver[edge[i].v].e_flow += flow;
                edge[i].flow += flow;
                updateReverseEdgeFlow(i, flow);
                return true;
            }
        }
    }
    return false;
}

__device__ void Graph::relabel(int u) {
    int mh = INT_MAX;
    for (int i = 0; i < edgeCount; i++) {
        if (edge[i].u == u) {
            if (edge[i].flow == edge[i].capacity)
                continue;
            if (ver[edge[i].v].h < mh) {
                mh = ver[edge[i].v].h;
                ver[u].h = mh + 1;
            }
        }
    }
}

__device__ int Graph::getMaxFlow(int s, int t) {
    preflow(s);
    while (true) {
        int u = overFlowVertex(ver, v);
        if (u == -1) break;
        if (!push(u)) {
            relabel(u);
        }
    }
    return ver[t].e_flow;
}

__global__ void calculate_graphs(Graph* graphs, int* results, int numInstances, int source, int sink,int num_of_threads) {
    int idx = threadIdx.x;
    
    for (int i = idx; i < numInstances; i += num_of_threads) {
        results[i] = graphs[i].getMaxFlow(source, sink);
       // printf("Max flow for graph %d: %d", i, results[i]);
    }

}


int main() {
    int V = 100;
    const int minGraphs = 500;
    const int maxGraphs = 5000;
    const double density = 0.25;
    int numGraphs=100;
    int minV = 10;
    int maxV = 100;
    int capacity = 1000;

    int number_of_threads = 1024;

    // Timing variables
    double start, stop;

  
        int source = 0, sink = V - 1;
        std::cout << "Running for " << numGraphs << " graphs..." << std::endl;

        // Generate random graphs
        Graph** h_graphs = new Graph * [numGraphs];
        srand(time(0)); // Seed for random number generation

        for (int i = 0; i < numGraphs; i++) {
            h_graphs[i] = generateRandomGraph(V, capacity, density);
        }

        start = omp_get_wtime();

        // Allocate memory on the device
        Graph* d_graphs;
        int* d_results;
        cudaMallocManaged(&d_graphs, numGraphs * sizeof(Graph));
        cudaMallocManaged(&d_results, numGraphs * sizeof(int));

        // Copy data from host to device
        for (int i = 0; i < numGraphs; i++) {
            cudaMallocManaged(&(d_graphs[i].ver), V * sizeof(Vertex));
            cudaMallocManaged(&(d_graphs[i].edge), 2 * density * V * (V - 1) * sizeof(Edge));
            cudaMemcpy(d_graphs[i].ver, h_graphs[i]->ver, V * sizeof(Vertex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_graphs[i].edge, h_graphs[i]->edge, 2 * density * V * (V - 1) * sizeof(Edge), cudaMemcpyHostToDevice);
            d_graphs[i].v = h_graphs[i]->v;
            d_graphs[i].edgeCount = h_graphs[i]->edgeCount;
        }

        // Launch CUDA kernel
        int blocks = (numGraphs + number_of_threads - 1) / number_of_threads;
        calculate_graphs << <1, number_of_threads >> > (d_graphs, d_results, numGraphs, source, sink, number_of_threads);
        cudaDeviceSynchronize();

        // Stop timing for GPU execution
        stop = omp_get_wtime();
        std::cout << "Time for GPU execution: " << (stop - start) << " s" << std::endl;

        // Copy results back to host
        int* h_results = new int[numGraphs];
        cudaMemcpy(h_results, d_results, numGraphs * sizeof(int), cudaMemcpyDeviceToHost);

        // Print the results
        for (int i = 0; i < numGraphs; i++) {
            std::cout << "Max flow for graph " << i << ": " << h_results[i] << std::endl;
        }

        // Free memory
        for (int i = 0; i < numGraphs; i++) {
            cudaFree(d_graphs[i].ver);
            cudaFree(d_graphs[i].edge);
            delete h_graphs[i];
        }
        delete[] h_graphs;
        cudaFree(d_graphs);
        cudaFree(d_results);

        std::cout << "Experiment completed for " << numGraphs << " graphs." << std::endl << std::endl;
    

    return 0;
}