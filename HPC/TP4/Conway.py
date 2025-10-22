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
    
    return np.int(init), np.int(fin)

def create_local_grid(grid):
    
    init, fin = idim_local(grid)
    
    sousgrid = grid[init:fin+1,:]
    
    ghostgrid = conway.enlarge_grid(sousgrid)
    
    if rank != size-1:    
        message_env = ghostgrid[fin+1,:]
        MPI.COMM_WORLD.send(message_env,dest=rank+1)
        
        message_rec=MPI.COMM_WORLD.recv(source=rank+1)
        ghostgrid[fin+1,:] = message_rec
        
        
    elif rank != 0:
        message_env = ghostgrid[init+1,:]
        MPI.COMM_WORLD.send(message_env,dest=rank-1)
        
        message_rec=MPI.COMM_WORLD.recv(source=rank-1)
        ghostgrid[init+1,:] = message_rec
        
    return ghostgrid
           
    
if (__name__ == "__main__"):
    
    grid = conway.init_grid((10,10))
    
    plt.imshow(grid,cmap=plt.cm.binary)
    
    sousgrid = create_local_grid(grid)
    
    if rank == 0:
        print(sousgrid)