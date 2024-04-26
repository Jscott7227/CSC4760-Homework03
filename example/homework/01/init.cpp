
#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
  
using namespace std;
int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int P = 2;  // Number of rows
    int Q = size / P;  // Number of columns
    
    // Calculate local size for each process
    int local_size = Q * P;
    int M = size;
    // Initialize vector x with length M
    vector<int> x(local_size);
    // Initialize vector y with length M and replicated horizontally
    vector<int> y(M);
    if (rank == 0) {
        for (int i = 0; i < M; ++i) {
            y[i] = i;
        }
    }
    // Scatter y vertically
    MPI_Scatter(&y[rank * local_size], local_size, MPI_INT, &x[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
  
    // Broadcast x horizontally in each process row
    MPI_Bcast(&x[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
    // Print out the received values
    cout << "Rank: " << rank << " x: ";
    for (int i = 0; i < local_size; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;
    MPI_Finalize();
    return 0;
}
