from mpi4py import MPI 
import numpy as np
import M2SD_HPC_TP04_MPIConway as conway
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def idim_local(grid):
    irange, jrange = grid.shape
    x = irange / size
    init = x * rank
    fin = init + x - 1
    if rank == size - 1:
        fin = irange
    return int(init), int(fin)


def create_local_grid(grid):
    init, fin = idim_local(grid)
    sousgrid = grid[init:fin + 1, :]
    ghostgrid = conway.enlarge_grid(sousgrid)

    up = rank - 1
    down = rank + 1

    reqs = []
    if down < size:
        reqs.append(comm.Isend(sousgrid[-1, :].copy(), dest=down))
        recv_bottom = np.empty_like(sousgrid[0, :])
        reqs.append(comm.Irecv(recv_bottom, source=down))
    else:
        recv_bottom = None

    if up >= 0:
        reqs.append(comm.Isend(sousgrid[0, :].copy(), dest=up))
        recv_top = np.empty_like(sousgrid[0, :])
        reqs.append(comm.Irecv(recv_top, source=up))
    else:
        recv_top = None

    MPI.Request.Waitall(reqs)

    if recv_top is not None:
        ghostgrid[0, 1:-1] = recv_top
    if recv_bottom is not None:
        ghostgrid[-1, 1:-1] = recv_bottom

    return ghostgrid


def conway_p2p(grid, epochs):
    total_start = MPI.Wtime()

    for ep in range(epochs):
        # Local grid creation + halo exchange
        local_grid = create_local_grid(grid)
        egrid = conway.life_step(local_grid)
        local_result = egrid[1:-1, 1:-1]
        gathered = comm.gather(local_result, root=0)

        if rank == 0:
            grid = np.vstack(gathered)

    total_end = MPI.Wtime()
    grid = comm.bcast(grid if rank == 0 else None, root=0)
    return grid, total_end - total_start


if __name__ == "__main__":
    grid_sizes = [(10, 10), (100, 100), (1000, 1000), (2000, 2000)]
    epochs = 5
    timings = []

    for gsize in grid_sizes:
        # Initialize grid on rank 0
        if rank == 0:
            grid = conway.init_grid(gsize, threshold=0.4)
        else:
            grid = None

        # Broadcast to all ranks
        grid = comm.bcast(grid, root=0)

        # Run Conway and measure total execution time
        _, exec_time = conway_p2p(grid, epochs)

        # Collect times from all ranks and take the max (slowest rank determines runtime)
        total_time = comm.reduce(exec_time, op=MPI.MAX, root=0)

        if rank == 0:
            timings.append((gsize[0], gsize[1], epochs, total_time))

    # Print results only on rank 0
    if rank == 0:
        print("\n=== Performance Results ===")
        print(f"{'Grid Size':>12} | {'Epochs':>6} | {'Execution Time (s)':>20}")
        print("-" * 45)
        for n, m, ep, t in timings:
            print(f"{str((n,m)):>12} | {ep:>6} | {t:>20.6f}")
        print("-" * 45)
