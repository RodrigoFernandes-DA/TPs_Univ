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
    
    up = rank - 1
    down = rank + 1

    # Send/receive using non-blocking to avoid deadlocks
    reqs = []

    # Send top row and receive bottom ghost
    if down < size:
        reqs.append(comm.Isend(sousgrid[-1, :].copy(), dest=down))
        recv_bottom = np.empty_like(sousgrid[0, :])
        reqs.append(comm.Irecv(recv_bottom, source=down))
    else:
        recv_bottom = None

    # Send bottom row and receive top ghost
    if up >= 0:
        reqs.append(comm.Isend(sousgrid[0, :].copy(), dest=up))
        recv_top = np.empty_like(sousgrid[0, :])
        reqs.append(comm.Irecv(recv_top, source=up))
    else:
        recv_top = None

    MPI.Request.Waitall(reqs)

    # Fill ghost zones
    if recv_top is not None:
        ghostgrid[0, 1:-1] = recv_top
    if recv_bottom is not None:
        ghostgrid[-1, 1:-1] = recv_bottom
        
    # print(f"rank = {rank} \n {ghostgrid}")

    return ghostgrid
   
           
def conway_p2p(grid, epochs):
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    for ep in range(epochs):
        # Cada processo cria sua grade local com bordas corretas
        local_grid = create_local_grid(grid)
        
        # Aplica o passo do Jogo da Vida localmente
        egrid = conway.life_step(local_grid)
        
        # Remove as bordas fantasmas
        local_result = egrid[1:-1, 1:-1]

        # Junta tudo no rank 0
        gathered = comm.gather(local_result, root=0)

        if rank == 0:
            # Reconstrói a grade global para a próxima época
            grid = np.vstack(gathered)

    # Broadcast final da grade completa para todos (opcional)
    grid = comm.bcast(grid if rank == 0 else None, root=0)
    return grid


def conway_coll(grid, epochs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    irange, jrange = grid.shape

    # Determine local sizes (balanced decomposition)
    counts = [irange // size + (1 if r < irange % size else 0) for r in range(size)]
    starts = [sum(counts[:r]) for r in range(size)]

    local_rows = counts[rank]
    local_grid = np.zeros((local_rows, jrange), dtype=grid.dtype)

    for ep in range(epochs):
        # Scatter subgrids
        sendbuf = None
        if rank == 0:
            sendbuf = [grid, tuple(np.array(counts) * jrange), tuple(np.array(starts) * jrange), MPI.DOUBLE]

        comm.Scatterv(sendbuf, local_grid, root=0)

        # Enlarge, step, crop
        egrid = conway.enlarge_grid(local_grid)
        egrid = conway.life_step(egrid)
        local_result = egrid[1:-1, 1:-1]

        # Gather results
        recvbuf = None
        if rank == 0:
            recvbuf = np.zeros_like(grid)
        comm.Gatherv(local_result, [recvbuf, tuple(np.array(counts) * jrange), tuple(np.array(starts) * jrange), MPI.DOUBLE], root=0)

        # Prepare grid for next iteration
        grid = comm.bcast(recvbuf if rank == 0 else None, root=0)

    return grid



if (__name__ == "__main__"):
    # exemplo de uso
    # criar grid inicial no rank 0 e broadcast para todos (ou cada rank pode gerar a mesma seed)
    if rank == 0:
        grid = conway.init_grid((6, 6), threshold=0.4)
    else:
        grid = None
    # broadcast the full global grid to all ranks so each rank can compute idim_local
    grid = comm.bcast(grid, root=0)

    if rank == 0:
        print("Grid inicial:")
        print(grid)

    # Test both versions
    if rank == 0:
        print("\n=== Testing collective version ===")
    
    result_coll = conway_coll(grid, epochs=2)

    if rank == 0:
        print("Resultado final (collective):")
        print(result_coll)

    if rank == 0:
        print("\n=== Testing point-to-point version ===")
    
    result_p2p = conway_p2p(grid, epochs=2)

    if rank == 0:
        print("Resultado final (point-to-point):")
        print(result_p2p)
        
        # Check if results are the same
        if np.array_equal(result_coll, result_p2p):
            print("\n✓ Both versions produced identical results!")
        else:
            print("\n✗ Results differ between versions!")
        
        # opcional: mostrar imagem
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(result_coll, cmap=plt.cm.binary)
        plt.title("Collective Version")
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_p2p, cmap=plt.cm.binary)
        plt.title("Point-to-Point Version")
        plt.show()
        
        
        
        
        
        
# if __name__ == "__main__":
#     # exemplo de uso
#     # criar grid inicial no rank 0 e broadcast para todos (ou cada rank pode gerar a mesma seed)
#     if rank == 0:
#         grid = conway.init_grid((6, 6), threshold=0.4)
#     else:
#         grid = None
    
#     # Broadcast the full global grid to all ranks so each rank can compute idim_local
#     grid = comm.bcast(grid, root=0)

#     if rank == 0:
#         print("Grid inicial:")
#         print(grid)

#     # Run both versions for comparison
#     if rank == 0:
#         print("\n=== Conway com Comunicação Coletiva ===")
    
#     start_time = time.time()
#     result_coll = conway_coll(grid, epochs=2)
#     coll_time = time.time() - start_time

#     if rank == 0:
#         print("Resultado final (comunicação coletiva):")
#         print(result_coll)
#         print(f"Tempo com comunicação coletiva: {coll_time:.6f} segundos")
        
#         # opcional: mostrar imagem
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(grid, cmap=plt.cm.binary)
#         plt.title("Grid Inicial")
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(result_coll, cmap=plt.cm.binary)
#         plt.title("Resultado Final (Coletiva)")
#         plt.show()