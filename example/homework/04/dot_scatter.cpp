#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric> // for std::inner_product

using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = size; // Length of the vectors
    int local_size = M / size; // Size of local portion of vectors

    // Initialize vectors
    vector<int> horizontal_vec(M); // Horizontal vector
    vector<int> vertical_vec(local_size); // Vertical vector

    // Linear load-balanced distribution for horizontal vector
    for (int i = 0; i < M; ++i) {
        horizontal_vec[i] = i + 1; // Just an example initialization
    }
    cout << "Horizontal Vector:" << endl;
    for (int i = 0; i < M; ++i) {
        cout << horizontal_vec[i] << " ";
    }
    cout << endl;

    // Scatter distribution for vertical vector
    MPI_Scatter(&horizontal_vec[0], local_size, MPI_INT, &vertical_vec[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print out the local portion of the vertical vector
    cout << "Rank: " << rank << " Vertical Vector:" << endl;
    for (int i = 0; i < local_size; ++i) {
        cout << vertical_vec[i] << " ";
    }
    cout << endl;

    // Broadcast the horizontal vector to all processes
    MPI_Bcast(&horizontal_vec[0], M, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the partial dot product
    int partial_dot_product = inner_product(vertical_vec.begin(), vertical_vec.end(), horizontal_vec.begin() + rank * local_size, 0);

    // Sum up the partial dot products from all processes
    int total_dot_product;
    MPI_Reduce(&partial_dot_product, &total_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print out the result
    if (rank == 0) {
        cout << "Dot product: " << total_dot_product << endl;
    }

    MPI_Finalize();
    return 0;
}