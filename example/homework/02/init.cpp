#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

class Domain {
public:
    Domain(int _M, int _N, int _halo_depth, const char* _name = "", char (*initial_values)[10] = nullptr) :
        M(_M), N(_N), halo_depth(_halo_depth), name(_name)
    {
        // Initialize Kokkos view for interior data
        interior = Kokkos::View<char**>("Interior View", M, N);

        // Initialize Kokkos view for exterior data with halos
        exterior = Kokkos::View<char*>("Exterior View", (M + 2 * halo_depth) * (N + 2 * halo_depth));

        // Copy initial values to the interior view if provided
        if (initial_values) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    interior(i, j) = initial_values[i][j];
                }
            }
        }
    }

    ~Domain() = default;

    // Accessors for M, N, and name
    int rows() const { return M; }
    int cols() const { return N; }
    const char* myname() const { return name; }

    // Accessors for Kokkos views
    Kokkos::View<char**> getInteriorView() const { return interior; }
    Kokkos::View<char*> getExteriorView() const { return exterior; }

private:
    int M;              // Number of rows
    int N;              // Number of columns
    int halo_depth;     // Depth of halos
    const char* name;   // Name of the domain

    Kokkos::View<char**> interior; // Kokkos view for interior data
    Kokkos::View<char*> exterior;  // Kokkos view for exterior data with halos
};

inline char update_the_cell(char cell, int neighbor_count) {
    char newcell;
    if (cell == 0)
        newcell = (neighbor_count == 3) ? 1 : 0;
    else
        newcell = ((neighbor_count == 2) || (neighbor_count == 3)) ? 1 : 0;
    return newcell;
}

void print_domain(Domain domain) {
    std::cout << domain.myname() << ":" << std::endl;

    auto interior = domain.getInteriorView();
    auto rows = domain.rows();
    auto cols = domain.cols();

    // Loop through the domain and print each element
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            char value = interior(i, j);
            std::cout << (value ? "*" : " ");
        }
        std::cout << std::endl;
    }
}

void update_domain(Domain& new_domain, Domain& old_domain) {
    auto interior_old = old_domain.getInteriorView();
    auto exterior_old = old_domain.getExteriorView();
    auto interior_new = new_domain.getInteriorView();
    auto exterior_new = new_domain.getExteriorView();

    int M = new_domain.rows();
    int N = new_domain.cols();
    int halo_depth = (exterior_new.extent(0) - interior_new.extent(0)) / 2;
    // Compute on the interior of the domain using Kokkos parallel_for
    Kokkos::parallel_for("update_domain", M, KOKKOS_LAMBDA(int i) {
        Kokkos::parallel_for(N, KOKKOS_LAMBDA(int j) {
            int neighbor_count = 0;
            for (int ni = i - 1; ni <= i + 1; ++ni) {
                for (int nj = j - 1; nj <= j + 1; ++nj) {
                    if (ni >= 0 && ni < M && nj >= 0 && nj < N) {
                        neighbor_count += interior_old(ni, nj);
                    } else {
                        neighbor_count += exterior_old[(ni + halo_depth) * (N + 2 * halo_depth) + (nj + halo_depth)];
                    }
                }
            }
            // Subtract the value of the current cell since it was included in the neighbor count
            neighbor_count -= interior_old(i, j);

            // Update the cell based on the neighbor count
            interior_new(i, j) = update_the_cell(interior_old(i, j), neighbor_count);
        });
    });
    print_domain(new_domain);
}


int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    // Create domain objects
    int M = 10;
    int N = 10;
    int halo_depth = 1;
    char initial_values[10][10] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    Domain* even_domain = new Domain(M, N, halo_depth, "even local domain", initial_values);
    Domain* odd_domain = new Domain(M, N, halo_depth, "odd local domain");

    // Perform domain updates
    for(int i = 0; i < 10; i++){
        update_domain(*odd_domain, *even_domain);
        update_domain(*even_domain, *odd_domain);
        }
    delete even_domain;
    delete odd_domain;
    Kokkos::finalize();
    return 0;
}
