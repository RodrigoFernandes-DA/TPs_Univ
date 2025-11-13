from mpi4py import MPI 
import numpy as np
import M2SD_HPC_TP04_MPIConway as base
import Conway_p2p as p2p
import Conway_coll as coll

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == '__main__':
    
    if rank == 0:
        grid = base.init_grid((5, 5), threshold=0.4)
        res = base.conway(grid, 1)
    else:
        grid = None
    
    
    # Broadcast to all ranks
    grid = comm.bcast(grid, root=0)
    res_p2p,_ = p2p.conway_p2p(grid, 1)
    res_coll,_ = coll.conway_coll(grid, 1)
    
    if rank == 0:
        print("Check result p2p : ",np.array_equal(res, res_p2p))
        print("Check result coll : ",np.array_equal(res, res_coll))
        print("Check result both : ",np.array_equal(res_p2p, res_coll))
        
        print(np.shape(res),"\n")
        print(np.shape(res_p2p),"\n")
        print(np.shape(res_coll))
        
        print(grid,"\n")
        print(res,"\n")
        print(res_p2p,"\n")
        print(res_coll,"\n")