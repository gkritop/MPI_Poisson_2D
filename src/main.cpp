#include "poisson.hpp"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static Options parse_args(int argc, char** argv) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--nx") && i+1 < argc) opt.nx = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ny") && i+1 < argc) opt.ny = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i+1 < argc) opt.tol = std::atof(argv[++i]);
        else if (!strcmp(argv[i], "--maxiter") && i+1 < argc) opt.maxiter = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--local")) opt.local_sizes = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            std::printf("Usage: %s [--nx N --ny N] [--local] [--tol T] [--maxiter M]\n", argv[0]);
            std::exit(0);
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm world = MPI_COMM_WORLD;

    int world_rank = 0, world_size = 1;

    MPI_Comm_rank(world, &world_rank);
    MPI_Comm_size(world, &world_size);

    const bool csv_only = std::getenv("POISSON2D_CSV") != nullptr;
    const bool verbose  = std::getenv("POISSON2D_VERBOSE") != nullptr && !csv_only;

    Options opt = parse_args(argc, argv);

    Grid g;
    setup_grid(g, opt, world);
    init_fields(g);

    if (verbose && g.rank == 0) {
        std::printf("[info] size=%d, px=%d py=%d, nxg=%d nyg=%d, nxl=%d nyl=%d, tol=%.2e, maxiter=%d\n",
               world_size, g.dims[0], g.dims[1], g.nx_global, g.ny_global,
               g.nx_local, g.ny_local, opt.tol, opt.maxiter);
    }

    const double t0 = MPI_Wtime();
    double res = 0.0;
    int it = 0;

    for (it = 1; it <= opt.maxiter; ++it) {
        exchange_halos(g);
        jacobi_step(g);

        // copy interior back
        for (int j = 1; j <= g.ny_local; ++j)
            for (int i = 1; i <= g.nx_local; ++i)
                g.u[g.idx(i,j)] = g.unew[g.idx(i,j)];
        res = compute_residual(g);

        if (res < opt.tol) break;

        if (verbose && g.rank == 0 && (it % 1000 == 0)) {
            std::printf("[iter] %d residual=%.6e\n", it, res);
        }
    }

    const double t1 = MPI_Wtime();
    const double dt = t1 - t0;

    if (g.rank == 0) {
        const char* mode = opt.local_sizes ? "local" : "global";

        std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6e\n",
            mode, world_size, g.dims[0], g.dims[1],
            g.nx_global, g.ny_global, g.nx_local, g.ny_local,
            it, dt, res);

        std::fflush(stdout);
    }

    if (g.cart != MPI_COMM_NULL) MPI_Comm_free(&g.cart);
    MPI_Finalize();
    
    return 0;
}
