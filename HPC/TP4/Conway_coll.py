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
    return grid

if __name__ == "__main__":
    # Example usage
    # Create initial grid on rank 0 and broadcast to all
    if rank == 0:
        grid = conway.init_grid((6, 6), threshold=0.4)
    else:
        grid = None
    
    # Broadcast the full global grid to all ranks
    grid = comm.bcast(grid, root=0)

    if rank == 0:
        print("Grid inicial:")
        print(grid)

    # Run the parallel Conway with collective communication
    result = conway_coll(grid, epochs=2)

    if rank == 0:
        print("Resultado final (recolhido no rank 0):")
        print(result)
        # Optional: show image
        plt.imshow(result, cmap=plt.cm.binary)
        plt.title("Resultado final com comunicação coletiva")
        plt.show()
