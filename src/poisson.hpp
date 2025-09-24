#pragma once
#include <vector>
#include <mpi.h>

struct Grid {
    MPI_Comm cart = MPI_COMM_NULL;

    int rank = 0, size = 1;
    int dims[2] = {0,0};
    int coords[2] = {0,0};

    int nbr_left = MPI_PROC_NULL, nbr_right = MPI_PROC_NULL;
    int nbr_up   = MPI_PROC_NULL, nbr_down  = MPI_PROC_NULL;

    int nx_global = 0, ny_global = 0; // interior sizes (global)
    int nx_local  = 0, ny_local  = 0; // interior sizes (per rank)
    
    double hx = 0.0, hy = 0.0;

    std::vector<double> u, unew, f;
    std::vector<double> sendL, sendR, recvL, recvR;

    inline int idx(int i, int j) const { return i + (nx_local + 2) * j; }
};

struct Options {
    int nx = 256;
    int ny = 256;
    int maxiter = 10000;
    double tol = 1e-6;
    bool local_sizes = false;
};

void setup_grid(Grid& g, const Options& opt, MPI_Comm world);
void init_fields(Grid& g);
void exchange_halos(Grid& g);
int  jacobi_step(Grid& g);
double compute_residual(Grid& g);

// Kept API tiny. Index math is simple.
