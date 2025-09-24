#include "poisson.hpp"
#include <cstdio>
#include <cmath>

static void create_cart(Grid& g, MPI_Comm world) {
    MPI_Comm_size(world, &g.size);
    MPI_Comm_rank(world, &g.rank);

    int dims[2] = {0,0};

    MPI_Dims_create(g.size, 2, dims);

    g.dims[0] = dims[0]; g.dims[1] = dims[1];

    int periods[2] = {0,0};

    MPI_Cart_create(world, 2, dims, periods, 1, &g.cart);
    MPI_Cart_coords(g.cart, g.rank, 2, g.coords);
    MPI_Cart_shift(g.cart, 0, 1, &g.nbr_left, &g.nbr_right);
    MPI_Cart_shift(g.cart, 1, 1, &g.nbr_up,   &g.nbr_down);
}

void setup_grid(Grid& g, const Options& opt, MPI_Comm world) {
    create_cart(g, world);

    if (opt.local_sizes) {
        g.nx_local  = opt.nx;
        g.ny_local  = opt.ny;
        g.nx_global = g.nx_local * g.dims[0];
        g.ny_global = g.ny_local * g.dims[1];
    } else {
        g.nx_global = opt.nx;
        g.ny_global = opt.ny;
        if (g.nx_global % g.dims[0] != 0 || g.ny_global % g.dims[1] != 0) {
            if (g.rank == 0) {
                std::fprintf(stderr, "[error] nx,ny must be divisible by px=%d py=%d\n", g.dims[0], g.dims[1]);
            }
            MPI_Abort(world, 1);
        }

        g.nx_local = g.nx_global / g.dims[0];
        g.ny_local = g.ny_global / g.dims[1];
    }

    g.hx = 1.0 / (g.nx_global + 1);
    g.hy = 1.0 / (g.ny_global + 1);

    const int nTot = (g.nx_local + 2) * (g.ny_local + 2);

    g.u.assign(nTot, 0.0);
    g.unew.assign(nTot, 0.0);
    g.f.assign(nTot, 1.0);

    g.sendL.resize(g.ny_local);
    g.sendR.resize(g.ny_local);
    g.recvL.resize(g.ny_local);
    g.recvR.resize(g.ny_local);
}

void init_fields(Grid& g) {
    // Zero initial guess; ghosts enforce Dirichlet 0.
}

void exchange_halos(Grid& g) {
    MPI_Status st;
    // Pack columns
    for (int j = 1; j <= g.ny_local; ++j) {
        g.sendL[j-1] = g.u[g.idx(1, j)];
        g.sendR[j-1] = g.u[g.idx(g.nx_local, j)];
    }

    // left<->right
    MPI_Sendrecv(g.sendL.data(), g.ny_local, MPI_DOUBLE, g.nbr_left, 100,
                 g.recvR.data(), g.ny_local, MPI_DOUBLE, g.nbr_right, 100,
                 g.cart, &st);
    MPI_Sendrecv(g.sendR.data(), g.ny_local, MPI_DOUBLE, g.nbr_right, 101,
                 g.recvL.data(), g.ny_local, MPI_DOUBLE, g.nbr_left, 101,
                 g.cart, &st);
    // Unpack into ghosts
    for (int j = 1; j <= g.ny_local; ++j) {
        g.u[g.idx(0, j)] = (g.nbr_left != MPI_PROC_NULL) ? g.recvL[j-1] : 0.0;
        g.u[g.idx(g.nx_local + 1, j)] = (g.nbr_right != MPI_PROC_NULL) ? g.recvR[j-1] : 0.0;
    }
    // Rows are contiguous
    MPI_Sendrecv(&g.u[g.idx(1,1)], g.nx_local, MPI_DOUBLE, g.nbr_up, 200,
                 &g.u[g.idx(1, g.ny_local + 1)], g.nx_local, MPI_DOUBLE, g.nbr_down, 200,
                 g.cart, &st);
    MPI_Sendrecv(&g.u[g.idx(1, g.ny_local)], g.nx_local, MPI_DOUBLE, g.nbr_down, 201,
                 &g.u[g.idx(1, 0)], g.nx_local, MPI_DOUBLE, g.nbr_up, 201,
                 g.cart, &st);

    if (g.nbr_up == MPI_PROC_NULL) for (int i = 1; i <= g.nx_local; ++i) g.u[g.idx(i,0)] = 0.0;
    if (g.nbr_down == MPI_PROC_NULL) for (int i = 1; i <= g.nx_local; ++i) g.u[g.idx(i,g.ny_local+1)] = 0.0;
    // Use MPI_Type_vector for columns.
}

int jacobi_step(Grid& g) {
    const double h2 = g.hx * g.hx; // assume hx==hy

    for (int j = 1; j <= g.ny_local; ++j) {
        for (int i = 1; i <= g.nx_local; ++i) {
            const double sum_nb = g.u[g.idx(i-1,j)] + g.u[g.idx(i+1,j)] +
                                  g.u[g.idx(i,j-1)] + g.u[g.idx(i,j+1)];
            g.unew[g.idx(i,j)] = 0.25 * (sum_nb + h2 * g.f[g.idx(i,j)]);
        }
    }

    return g.nx_local * g.ny_local;
}

double compute_residual(Grid& g) {
    const double inv_h2 = 1.0 / (g.hx * g.hx);
    double local_sum2 = 0.0;
    
    for (int j = 1; j <= g.ny_local; ++j) {
        for (int i = 1; i <= g.nx_local; ++i) {
            const double lap = ( g.u[g.idx(i-1,j)] + g.u[g.idx(i+1,j)]
                               + g.u[g.idx(i,j-1)] + g.u[g.idx(i,j+1)]
                               - 4.0 * g.u[g.idx(i,j)] ) * inv_h2;

            const double r = g.f[g.idx(i,j)] + lap;
            local_sum2 += r * r;
        }
    }

    double global_sum2 = 0.0;
    MPI_Allreduce(&local_sum2, &global_sum2, 1, MPI_DOUBLE, MPI_SUM, g.cart);
    const double n_tot = static_cast<double>(g.nx_global) * static_cast<double>(g.ny_global);
    
    return std::sqrt(global_sum2 / n_tot);
}
// Add norm history output for prettier plots.
