# Conway.py
from mpi4py import MPI
import numpy as np
import sys
import time
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

def create_local_grid_coll(grid):
    """
    Create local grid with ghost cells using collective communication (MPI_Sendrecv)
    """
    start, end = idim_local(grid)

    # slice using end as exclusive
    if end <= start:  # this rank has zero rows
        ncols = grid.shape[1]
        # Return a minimal ghostgrid (2 rows for top/bottom ghosts + 2 cols for padding)
        return np.zeros((2, ncols + 2), dtype=grid.dtype)

    local_subgrid = grid[start:end, :]
    local_nrows, ncols = local_subgrid.shape

    # Create enlarged grid with ghost cells
    ghostgrid = conway.enlarge_grid(local_subgrid)

    # neighbor ranks
    up = rank - 1
    down = rank + 1

    # Prepare send/recv buffers for each neighbor (only if neighbor exists)
    if up >= 0:
        send_top = local_subgrid[0, :].copy()
        recv_top = np.empty_like(send_top)
        # exchange with up: send our top to up, receive up's bottom into recv_top
        comm.Sendrecv(send_top, dest=up, recvbuf=recv_top, source=up)
    else:
        recv_top = np.zeros(ncols, dtype=local_subgrid.dtype)

    if down < size:
        send_bottom = local_subgrid[-1, :].copy()
        recv_bottom = np.empty_like(send_bottom)
        # exchange with down: send our bottom to down, receive down's top into recv_bottom
        comm.Sendrecv(send_bottom, dest=down, recvbuf=recv_bottom, source=down)
    else:
        recv_bottom = np.zeros(ncols, dtype=local_subgrid.dtype)

    # Fill ghost zones (skip the corner columns)
    ghostgrid[0, 1:-1] = recv_top
    ghostgrid[-1, 1:-1] = recv_bottom
        
    return ghostgrid

def conway_coll(grid, epochs):
    """
    Conway's Game of Life using collective communications for ghost cell exchange
    """
    total_start = MPI.Wtime()

    for ep in range(epochs):
        # Each process creates its local grid with ghost cells
        local_grid = create_local_grid_coll(grid)

        # Apply life step locally (operate on enlarged grid)
        egrid = conway.life_step(local_grid)

        # Remove ghost cells
        local_result = egrid[1:-1, 1:-1]

        # Gather all results at root process
        gathered = comm.gather(local_result, root=0)

        if rank == 0:
            # Reconstruct global grid for next epoch
            # gathered is a list of arrays (some arrays may be shape (0,ncols))
            grid = np.vstack(gathered)

        # Broadcast updated grid to all ranks so next epoch uses the new grid
        grid = comm.bcast(grid if rank == 0 else None, root=0)

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
        _, exec_time = conway_coll(grid, epochs)

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
