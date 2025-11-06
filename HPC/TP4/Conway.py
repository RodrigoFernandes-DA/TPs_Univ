from mpi4py import MPI
import numpy as np
import sys
import time
import M2SD_HPC_TP04_MPIConway as conway
import matplotlib.pyplot as plt

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()


def  idim_local(grid):
    "en fonction du rang du processus retourne le nombre de ligne locale à traiter ainsi que la position (coordonnée de la 1ère ligne) de la sous-grille dans la grille globale"
    
    irange, jrange = grid.shape
    
    x = irange/size
    
    init = x * rank
    fin = init + x -1
    
    if rank == size -1:
        fin = irange
    
    return int(init), int(fin)

def create_local_grid(grid):
    
    init, fin = idim_local(grid)
    
    sousgrid = grid[init:fin+1,:]
    
    ghostgrid = conway.enlarge_grid(sousgrid)
    
    # up = rank - 1
    # down = rank + 1

    # # Send/receive using non-blocking to avoid deadlocks
    # reqs = []

    # # Send top row and receive bottom ghost
    # if down < size:
    #     reqs.append(comm.Isend(sousgrid[-1, :].copy(), dest=down))
    #     recv_bottom = np.empty_like(sousgrid[0, :])
    #     reqs.append(comm.Irecv(recv_bottom, source=down))
    # else:
    #     recv_bottom = None

    # # Send bottom row and receive top ghost
    # if up >= 0:
    #     reqs.append(comm.Isend(sousgrid[0, :].copy(), dest=up))
    #     recv_top = np.empty_like(sousgrid[0, :])
    #     reqs.append(comm.Irecv(recv_top, source=up))
    # else:
    #     recv_top = None

    # MPI.Request.Waitall(reqs)

    # # Fill ghost zones
    # if recv_top is not None:
    #     ghostgrid[0, 1:-1] = recv_top
    # if recv_bottom is not None:
    #     ghostgrid[-1, 1:-1] = recv_bottom
    
    if rank < size:    
        message_env = sousgrid[-1,:]
        MPI.COMM_WORLD.send(message_env,dest=rank+1)
        
        message_rec=MPI.COMM_WORLD.recv(source=rank+1)
        ghostgrid[fin+1,:] = message_rec
        
        
    elif rank != 0:
        message_env = ghostgrid[init+1,:]
        MPI.COMM_WORLD.send(message_env,dest=rank-1)
        
        message_rec=MPI.COMM_WORLD.recv(source=rank-1)
        ghostgrid[init+1,:] = message_rec
        
    print(f"rank = {rank} \n {ghostgrid}")

    return ghostgrid
           
    
if (__name__ == "__main__"):
    
    grid = conway.init_grid((5,5))
    
    plt.imshow(grid,cmap=plt.cm.binary)
    
    sousgrid = create_local_grid(grid)
    
    if rank == 0:
        print(grid)
        print(sousgrid)