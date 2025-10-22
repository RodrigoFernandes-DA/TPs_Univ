from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np

def pick(n):
    count_inside = 0
    for i in range(n):
        x, y = np.random.random(2) * 2 - 1
        if x**2 + y**2 <= 1:
            count_inside += 1
    return count_inside

def par_pick(n):
    nlocal = int(n / size)
    localcount = pick(nlocal)
    
    # Utilisation d'une communication collective (gather)
    counts = comm.gather(localcount, root=0)
    
    if rank == 0:
        count = sum(counts)
    else:
        count = None
    
    return count

if (__name__ == "__main__"):
    import sys
    import time
    n = int(sys.argv[1])
    tic = time.time()
    count = par_pick(n)
    if rank == 0:
        pmcpi = 4 * count / n
        toc = time.time()
        print(f'For {n} picks, the approximation of pi is {pmcpi} ({abs(pmcpi-np.pi)/np.pi*100:.2f}%) computed in {toc - tic:.2f} seconds')
