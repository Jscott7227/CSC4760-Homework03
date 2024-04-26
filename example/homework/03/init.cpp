#include <iostream>

#include <vector>

#include <mpi.h>



int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int P = 2; // Number of rows

    int Q = size / P; // Number of columns

    if (size != P * Q) {

        if (rank == 0)

            std::cout << "Number of processes is not compatible with the topology." << std::endl;

        MPI_Finalize();

        exit(0);

    }

    int p = rank / Q, q = rank % Q;

    int M = 15; // Length of vector x

    std::vector<int> xglobal, x, y;

    if (q == 0 && p == 0) {

        xglobal.resize(M);

        for (int i = 0; i < M; i++)

            xglobal[i] = i + 1;

    }

    int m = M / P + ((p < (M % P)) ? 1 : 0);

    x.resize(m);

    MPI_Comm col_comm;

    MPI_Comm_split(MPI_COMM_WORLD, q, p, &col_comm);

    if (q == 0) {

        std::vector<int> recvcounts(P), displs(P);

        for (int i = 0; i < P; i++) {

            recvcounts[i] = M / P;

            if (i < (M % P)) recvcounts[i]++;

            displs[i] = i * (M / P) + ((i < M % P) ? i : M % P);

        }

        MPI_Scatterv((rank == 0) ? xglobal.data() : nullptr, recvcounts.data(),

                     displs.data(), MPI_INT, x.data(), m, MPI_INT, 0, col_comm);

    }

    MPI_Comm row_comm;

    MPI_Comm_split(MPI_COMM_WORLD, p, q, &row_comm);

    MPI_Bcast(x.data(), m, MPI_INT, 0, row_comm);

    int n =  M / Q + ((q < (M % Q)) ? 1 : 0);

    y.resize(n);

    for (int j = 0; j < n; j++)

        y[j] = 0;

    int nominal1 = M / P; int extra1 = M % P;

    for (int i = 0; i < m; i++) {

        int I = i + ((p < extra1) ? (nominal1+1)*p : (extra1*(nominal1+1)+(p-extra1)*nominal1));

        int qhat = I % Q;

        int jhat = I / Q;

        if (qhat == q)

            y[jhat] = x[i];

    }

    MPI_Allreduce(MPI_IN_PLACE, y.data(), n, MPI_INT, MPI_SUM, col_comm);

    std::cout << "Process (" << p << "," << q << "): y = ";

    for (int i = 0; i < n; i++)

        std::cout << y[i] << " ";

    std::cout << std::endl;

    MPI_Comm_free(&row_comm);

    MPI_Comm_free(&col_comm);

    MPI_Finalize();

}


