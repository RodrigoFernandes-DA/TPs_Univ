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
    """
    Retorna (start, end) com end exclusivo -> slice grid[start:end, :]
    Distribui as linhas de forma equilibrada; primeiros 'remainder' ranks recebem uma linha extra.
    """
    nrows = grid.shape[0]
    base = nrows // size
    rem = nrows % size

    if rank < rem:
        local_n = base + 1
        start = rank * local_n
    else:
        local_n = base
        start = rem * (base + 1) + (rank - rem) * base

    end = start + local_n
    return start, end

def create_local_grid_coll(grid):
    """
    Create local grid with ghost cells using collective communication (MPI_Sendrecv)
    """
    init, fin = idim_local(grid)
    
    # Extract local subgrid without ghost cells
    local_subgrid = grid[init:fin+1, :]
    local_nrows, ncols = local_subgrid.shape
    
    # Create enlarged grid with ghost cells
    ghostgrid = conway.enlarge_grid(local_subgrid)
    
    # Determine neighbor ranks
    up = rank - 1
    down = rank + 1
    
    # Prepare send and receive buffers
    send_top = local_subgrid[0, :].copy() if up >= 0 else None
    send_bottom = local_subgrid[-1, :].copy() if down < size else None
    
    recv_top = np.empty(ncols, dtype=local_subgrid.dtype) if up >= 0 else None
    recv_bottom = np.empty(ncols, dtype=local_subgrid.dtype) if down < size else None
    
    # Use Sendrecv for non-blocking communication that avoids deadlocks
    # Send bottom row, receive top ghost row
    if down < size and up >= 0:
        comm.Sendrecv(send_bottom, dest=down, recvbuf=recv_top, source=up)
    elif down < size:
        comm.Send(send_bottom, dest=down)
    elif up >= 0:
        comm.Recv(recv_top, source=up)
    
    # Send top row, receive bottom ghost row  
    if up >= 0 and down < size:
        comm.Sendrecv(send_top, dest=up, recvbuf=recv_bottom, source=down)
    elif up >= 0:
        comm.Send(send_top, dest=up)
    elif down < size:
        comm.Recv(recv_bottom, source=down)
    
    # Fill ghost zones
    if recv_top is not None:
        ghostgrid[0, 1:-1] = recv_top
    if recv_bottom is not None:
        ghostgrid[-1, 1:-1] = recv_bottom
        
    return ghostgrid

def conway_coll(grid, epochs):
    """
    Conway's Game of Life using collective communications for ghost cell exchange
    """
    total_start = MPI.Wtime()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for ep in range(epochs):
        # Each process creates its local grid with ghost cells using collective communication
        local_grid = create_local_grid_coll(grid)
        
        # Apply life step locally
        egrid = conway.life_step(local_grid)
        
        # Remove ghost cells
        local_result = egrid[1:-1, 1:-1]

        # Gather all results at root process
        gathered = comm.gather(local_result, root=0)

        if rank == 0:
            # Reconstruct global grid for next epoch
            grid = np.vstack(gathered)

    # Broadcast final grid to all processes
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
