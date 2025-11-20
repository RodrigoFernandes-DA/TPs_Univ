from mpi4py import MPI 
import numpy as np
import M2SD_HPC_TP04_MPIConway as base
import Conway_p2p as p2p
import Conway_coll as coll
import Conway_block as block

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == '__main__':
    
    if rank == 0:
        # grid = read_grid('jdv_1M.grid')
        grid = base.init_grid((50, 50), threshold=0.4)
        res = base.conway(grid, 5)
    else:
        grid = None
    
    # Broadcast to all ranks
    grid = comm.bcast(grid, root=0)
    res_p2p,_ = p2p.conway_p2p(grid, 5)
    res_coll,_ = coll.conway_coll(grid, 5)
    res_block,_ = block.conway_block(grid, 5)
    
    if rank == 0:
        print("Check result p2p : ",np.array_equal(res, res_p2p))
        print("Check result coll : ",np.array_equal(res, res_coll))
        print("Check result block : ",np.array_equal(res, res_block))
        
        print(np.shape(res_block))
        
        print(res_block)