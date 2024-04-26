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

    int P = 2;
    int Q = size / P;

    int local_size = Q;
    int M = size;

    vector<int> x(local_size);
    vector<int> y(M);

    int J = rank;
    int j = J / Q;
    int q = J % Q;

    int scatter_local_size = M / Q;

    if (rank == 0) {
        for (int i = 0; i < M; ++i) {
            y[i] = i;
        }
    }

    MPI_Scatter(&y[q * scatter_local_size], scatter_local_size, MPI_INT, &x[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Rank: " << rank << " x: ";
    for (int i = 0; i < local_size; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;

    MPI_Finalize();
    return 0;
}
