#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);  
	Kokkos::initialize(argc, argv);
  	{
		using namespace std;
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>

#include <mpi.h>

// forward declarations:

class Domain
{
public:
  Domain(int _M, int _N, const char *_name="") : domain(_M, vector<char>(_N)), M(_M), N(_N), name(_name) {}
    virtual ~Domain() {}
    char& operator()(int i, int j) { return domain[i][j]; }
    char operator()(int i, int j) const { return domain[i][j]; }

  int rows() const {return M;}
  int cols() const {return N;}

  const string & myname() const {return name;}

protected:
  vector<vector<char>> domain; 
  int M;
  int N;
  string name;
};

void zero_domain(Domain &domain);
void print_domain(Domain &domain);
void update_domain(Domain &new_domain, Domain &old_domain, int size, int myrank, MPI_Comm comm);
void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm);

int main(int argc, char **argv)
{
  int M, N;
  int iterations;

  if(argc < 4)
  {
    cout << "usage: " << argv[0] << " M N iterations" << endl;
    exit(0);
  }

  int size, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int array[3];
  if(myrank == 0)
  {
     M = atoi(argv[1]); N = atoi(argv[2]); iterations = atoi(argv[3]);

     array[0] = M;
     array[1] = N;
     array[2] = iterations;
     
  }
  MPI_Bcast(array, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank != 0)
  {
    M = array[0];
    N = array[1];
    iterations = array[2];
  }

  
  parallel_code(M, N, iterations, size, myrank, MPI_COMM_WORLD);
  
  MPI_Finalize();
}

void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm)
{
  int m = M / size; // perfect divisibility for this version
  int n = N / size; // 2D decomposition
  
  Domain even_domain(m,n,"even Domain");
  Domain odd_domain(m,n,"odd Domain");

  zero_domain(even_domain);
  zero_domain(odd_domain);

  // fill in even_domain with something meaningful (initial state)
  // this requires min size for default values to fit:
  if((n >= 8) && (m >= 10))
  {
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;
    
    even_domain(3,5) = 1;
    even_domain(3,6) = 1;
    even_domain(3,7) = 1;

    even_domain(6,7) = 1;
    even_domain(7,7) = 1;
    even_domain(8,7) = 1;
    even_domain(9,7) = 1;
  }

  // here is where I might print out my picture of the initial domain
  cout << "Initial:"<<endl; print_domain(even_domain);

  Domain *odd, *even; // pointer swap magic
  odd = &odd_domain;
  even = &even_domain;

  for(int i = 0; i < iterations; ++i)
  {
    update_domain(*odd, *even, size, myrank, comm);
    // here is where I might print out my picture of the updated domain
    cout << "Iteration #" << i << endl; print_domain(*odd);

    // swap pointers:
    Domain *temp = odd;
    odd  = even;
    even = temp;
  }


}

void zero_domain(Domain &domain)
{
  for(int i = 0; i < domain.rows(); ++i)
    for(int j = 0; j < domain.cols(); ++j)
      domain(i,j) = 0;
}

void print_domain(Domain &domain)
{
  cout << domain.myname() << ":" <<endl;
  // this is naive; it doesn't understand big domains at all 
  for(int i = 0; i < domain.rows(); ++i)
  {
    for(int j = 0; j < domain.cols(); ++j)
      cout << (domain(i,j) ? "*" : " ");
    cout << endl;
  }
}

void update_domain(Domain& new_domain, Domain& old_domain, int size, int myrank, MPI_Comm comm) {
    MPI_Request request[8];
    int neighbor_count;
    int m = new_domain.rows();
    int n = new_domain.cols();

    char* top_row = new char[n];
    char* bottom_row = new char[n];
    char* left_column = new char[m];
    char* right_column = new char[m];

    // Sending and receiving data for neighboring rows and columns
    MPI_Isend(&old_domain(0, 0), n, MPI_CHAR, (myrank - 1 + size) % size, 0, comm, &request[0]);
    MPI_Isend(&old_domain(m - 1, 0), n, MPI_CHAR, (myrank + 1) % size, 0, comm, &request[1]);
    MPI_Isend(&old_domain(0, 0), 1, MPI_CHAR, (myrank - size + size) % size, 0, comm, &request[2]);
    MPI_Isend(&old_domain(0, n - 1), 1, MPI_CHAR, (myrank + size) % size, 0, comm, &request[3]);

    MPI_Irecv(top_row, n, MPI_CHAR, (myrank - 1 + size) % size, 0, comm, &request[4]);
    MPI_Irecv(bottom_row, n, MPI_CHAR, (myrank + 1) % size, 0, comm, &request[5]);
    MPI_Irecv(left_column, m, MPI_CHAR, (myrank - size + size) % size, 0, comm, &request[6]);
    MPI_Irecv(right_column, m, MPI_CHAR, (myrank + size) % size, 0, comm, &request[7]);

    MPI_Waitall(8, request, MPI_STATUSES_IGNORE);

    // Updating internal cells
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            neighbor_count = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0)
                        continue;  // Skip the current cell
                    // Calculate the coordinates of the neighbor, handling wrapping
                    int ni = (i + di + m) % m;
                    int nj = (j + dj + n) % n;
                    // If the neighbor is alive, increment the neighbor count
                    if (old_domain(ni, nj))
                        ++neighbor_count;
                }
            }
            // Adjust neighbor count based on edge conditions
            if (i == 0 && top_row[j])
                ++neighbor_count;
            if (i == m - 1 && bottom_row[j])
                ++neighbor_count;
            if (j == 0 && left_column[i])
                ++neighbor_count;
            if (j == n - 1 && right_column[i])
                ++neighbor_count;
            // Apply Game of Life rules
            char mycell = old_domain(i, j);
            char newcell = 0;
            if (mycell == 0)
                newcell = (neighbor_count == 3) ? 1 : 0;
            else
                newcell = ((neighbor_count == 2) || (neighbor_count == 3)) ? 1 : 0;
            // Update the new domain
            new_domain(i, j) = newcell;
        }
    }

    // Cleaning up
    delete[] top_row;
    delete[] bottom_row;
    delete[] left_column;
    delete[] right_column;
}
  	}
  	Kokkos::finalize();
	MPI_Finalize();
}
