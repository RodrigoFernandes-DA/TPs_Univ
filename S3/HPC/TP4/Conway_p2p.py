from mpi4py import MPI 
import numpy as np
import M2SD_HPC_TP04_MPIConway as conway
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def idim_local(grid):
    irange, jrange = grid.shape
    base = irange // size
    rest = irange % size

    # start index for this rank
    start = rank * base + min(rank, rest)
    # number of rows for this rank
    nrows = base + (1 if rank < rest else 0)
    end = start + nrows
    return int(start), int(end)


def create_local_grid(grid):
    start, end = idim_local(grid)
    # slice using end as exclusive
    if end <= start:
        # no rows for this rank -> create an empty sousgrid with correct number of columns
        jdim = grid.shape[1]
        sousgrid = np.zeros((0, jdim), dtype=grid.dtype)
    else:
        sousgrid = grid[start:end, :]

    # If a rank has zero rows, enlarge_grid must still return a valid enlarged array.
    # In that case we create an enlarged grid with two extra rows (top & bottom) and two extra cols.
    if sousgrid.size == 0:
        # make an enlarged ghostgrid with only the padding (2 x jdim+2)
        jdim = grid.shape[1]
        ghostgrid = np.zeros((2, jdim + 2), dtype=grid.dtype)
        # no halo exchange required because there is no local data, just zeros
        return ghostgrid

    ghostgrid = conway.enlarge_grid(sousgrid)

    up = rank - 1
    down = rank + 1
    top_row = sousgrid[0, :].copy()
    bottom_row = sousgrid[-1, :].copy()
    recv_top = np.empty_like(top_row)
    recv_bottom = np.empty_like(bottom_row)

    reqs = []
    if up >= 0:
        reqs.append(comm.Isend(top_row, dest=up, tag=11))
        reqs.append(comm.Irecv(recv_top, source=up, tag=22))
    else:
        recv_top = np.zeros_like(top_row)

    if down < size:
        reqs.append(comm.Isend(bottom_row, dest=down, tag=22))
        reqs.append(comm.Irecv(recv_bottom, source=down, tag=11))
    else:
        recv_bottom = np.zeros_like(bottom_row)

    MPI.Request.Waitall(reqs)

    ghostgrid[0, 1:-1] = recv_top
    ghostgrid[-1, 1:-1] = recv_bottom

    return ghostgrid


def conway_p2p(grid, epochs):
    total_start = MPI.Wtime()

    for ep in range(epochs):
        local_grid = create_local_grid(grid)
        egrid = conway.life_step(local_grid)
        local_result = egrid[1:-1, 1:-1]
        gathered = comm.gather(local_result, root=0)

        if rank == 0:
            grid = np.vstack(gathered)
        grid = comm.bcast(grid, root=0)

    total_end = MPI.Wtime()
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
