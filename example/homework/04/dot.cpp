#include <iostream>

#include <vector>

#include <mpi.h>



int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int P = 2;

    int Q = size / P;

    if (size != P * Q) {

        if (rank == 0)

            std::cout << "Number of processes is not compatible with the topology." << std::endl;

        MPI_Finalize();

        exit(0);

    }

    int p = rank / Q, q = rank % Q;

    int M = 15;

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

    for(int j = 0; j < n; j++)

        y[j] = 0;

    int nominal1 = M / P; int extra1 = M % P;

    int nominal2 = M / Q; int extra2 = M % Q;

    for(int i = 0; i < m; i++) {

        int I = i + ((p < extra1) ? (nominal1+1)*p :

                     (extra1*(nominal1+1)+(p-extra1)*nominal1));

        int qhat = (I < extra2*(nominal2+1)) ? I/(nominal2+1) : 

                   (extra2+(I-extra2*(nominal2+1))/nominal2);

        int jhat = I - ((qhat < extra2) ? (nominal2+1)*qhat :

                        (extra2*(nominal2+1) + (qhat-extra2)*nominal2));

        if(qhat == q) {

            y[jhat] = x[i];

        }

    }

    MPI_Allreduce(MPI_IN_PLACE, y.data(), n, MPI_INT, MPI_SUM, col_comm);



    // Print vectors x and y

    for (int i = 0; i < size; i++) {

        if (rank == i) {

            std::cout << "Process (" << p << "," << q << "): x = ";

            for (int j = 0; j < m; j++)

                std::cout << x[j] << " ";

            std::cout << std::endl;



            std::cout << "Process (" << p << "," << q << "): y = ";

            for (int j = 0; j < n; j++)

                std::cout << y[j] << " ";

            std::cout << std::endl;

        }

        MPI_Barrier(MPI_COMM_WORLD);

    }



    // Compute the dot product locally

    int local_dot = 0;

    for (int i = 0; i < n; i++) {

        local_dot += y[i] * y[i];

    }



    int global_dot;

    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);



    if (rank == 0) {

        std::cout << "Dot product of vectors: " << global_dot << std::endl;

    }



    MPI_Comm_free(&row_comm);

    MPI_Comm_free(&col_comm); 

    MPI_Finalize();

}


