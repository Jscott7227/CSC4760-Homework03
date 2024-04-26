

#include <iostream>



#include <vector>



#include <numeric>



#include <mpi.h>







void printMatrix(const std::vector<std::vector<int>>& matrix) {



    for (const auto& row : matrix) {



        for (int element : row) {



            std::cout << element << " ";



        }



        std::cout << std::endl;



    }



}







int main(int argc, char **argv) {



    MPI_Init(&argc, &argv);



    int rank, size;



    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    MPI_Comm_size(MPI_COMM_WORLD, &size);







    int M = 6;



    int N = 6;



    int P = size;



    int Q = N / P;



    std::vector<std::vector<int>> A(M / P, std::vector<int>(N, 0));



    std::vector<int> x(N);



    std::vector<int> y(M / P, 0);







    if (rank == 0) {



        for (int i = 0; i < N; ++i) {



            x[i] = i + 1;



        }



        std::cout << "Matrix x:" << std::endl;



        for (int i = 0; i < N; ++i) {



            std::cout << x[i] << " ";



        }



        std::cout << std::endl << std::endl;



    }







    MPI_Bcast(x.data(), N, MPI_INT, 0, MPI_COMM_WORLD);







    for (int i = 0; i < M / P; ++i) {



        for (int j = 0; j < N; ++j) {



            A[i][j] = (rank * M / P) + i;



        }



    }







    std::cout << "Matrix A before multiplication in process " << rank << ":" << std::endl;



    printMatrix(A);







    for (int i = 0; i < M / P; ++i) {



        for (int j = 0; j < N; ++j) {



            y[i] += A[i][j] * x[j];



        }



    }







    std::partial_sum(y.begin(), y.end(), y.begin());







    std::vector<int> recv_counts(P, M / P);



    std::vector<int> displacements_result(P, 0);



    for (int i = 1; i < P; ++i) {



        displacements_result[i] = displacements_result[i - 1] + (M / P);



    }



    std::vector<int> result(M);



    MPI_Gatherv(y.data(), M / P, MPI_INT, result.data(), recv_counts.data(), displacements_result.data(), MPI_INT, 0, MPI_COMM_WORLD);







    if (rank == 0) {



        std::cout << "Result matrix after multiplication:" << std::endl;



        for (int i = 0; i < M; ++i) {



            std::cout << result[i] << " ";



        }



        std::cout << std::endl;



    }







    MPI_Finalize();



    return 0;



}


