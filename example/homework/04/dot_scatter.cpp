#include <iostream>

#include <vector>

#include <mpi.h>



int main(int argc, char *argv[])

{

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int P = 2; // Number of rows

    int Q = size / P; // Number of columns

    // Check if the number of processes is compatible with the topology 

    if (size != P * Q)

    {

        if (rank == 0)

            std::cout << "Number of processes is not compatible with the topology." << std::endl;

        MPI_Finalize();

        exit(0);

    }

    //Your process needs coordinates:

    int p = rank / Q, q = rank % Q;



    //std::cout << "( " << p << "," << q << ") progressing..." << std::endl;

    int M = 15; // Length of vector x --- let's make it longer; M>=P always a must fyi.

    // Allocate memory for vector xglobal, x, and y

    std::vector<int> xglobal, x, y;

    // Initialize data only in process (0, 0)

    if (q == 0)

    {

        if (p == 0)

        {

            xglobal.resize(M);

            for (int i = 0; i < M; i++)

            {

                xglobal[i] = i + 1; // Initialize xglobal with some data

            }

            std::cout << "(0,0) filled in xglobal[]" << std::endl;

        }

    }

    //std::cout << "( " << p << "," << q << ") progressing more..." << std::endl;

    //all processes do these steps:

    int m = M / P + ((p < (M % P)) ? 1 : 0);

    x.resize(m); // local vector in each process

    //y.resize(M / P + ((q < M % Q) ? 1 : 0));



    std::cout << "( " << p << "," << q << ") m=" << m << std::endl;

    MPI_Comm col_comm;

    MPI_Comm_split(MPI_COMM_WORLD, q /*rank / Q*/, p /*sort key */, &col_comm);

    if (q == 0)

    {

        // Scatter data down the first column

        std::vector<int> recvcounts(P), displs(P);

        // this is really only needed in p==0, but not going to specialize now.

        for (int i = 0; i < P; i++)

        {

            recvcounts[i] = M / P;

            if (i < (M % P))

                recvcounts[i]++;

            displs[i] = i * (M / P) + ((i < M % P) ? i : M % P);

        }

        // the q==0 column scatters.

        MPI_Scatterv((rank == 0) ? xglobal.data() : nullptr, recvcounts.data(),

                     displs.data(), MPI_INT, x.data(), m, MPI_INT, 0, col_comm);

    }

    // Broadcast data horizontally in each process row

    MPI_Comm row_comm;

    MPI_Comm_split(MPI_COMM_WORLD, p /*rank / Q*/, q /*sort key */, &row_comm);

    MPI_Bcast(x.data(), m, MPI_INT, 0, row_comm); // replicates x in each proc. column



    // Moved this line down, and modified, since unrelated to first step above!

    int n = M / Q + ((q < (M % Q)) ? 1 : 0);

    y.resize(n); // REVISED.  Row replicated, Col distr.

    for (int j = 0; j < n; j++)

        y[j] = 0;



    int nominal1 = M / P;

    int extra1 = M % P;

    for (int i = 0; i < m; i++) // m is the local size of the vector x[]

    {

        // x local to global: given that this element is (p,i), what is its global index I?

        int I = i + ((p < extra1) ? (nominal1 + 1) * p :

                     (extra1 * (nominal1 + 1) + (p - extra1) * nominal1));

        // so to what (qhat,jhat) does this element of the original global vector go?

        int qhat = I % Q;

        int jhat = I / Q;

        if (qhat == q) // great, this process has an element of y!

        {

            y[jhat] = x[i];

        }

    }



    MPI_Allreduce(MPI_IN_PLACE, y.data(), n, MPI_INT, MPI_SUM, col_comm);



    // Dot product of y with itself

    int local_dot = 0;

    for (int i = 0; i < n; i++) {

        local_dot += y[i] * y[i];

    }

    int global_dot;

    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);



    if (rank == 0) {

        std::cout << "Dot product of y with itself: " << global_dot << std::endl;

    }



    // Finalize MPI

    MPI_Comm_free(&row_comm);

    MPI_Comm_free(&col_comm);

    MPI_Finalize();

}


