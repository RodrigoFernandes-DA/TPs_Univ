from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
import numpy as np
def pick(n):
    count_inside = 0
    for i in range(n):
        x, y = np.random.random(2) * 2 - 1
        if x**2 + y**2 <= 1: count_inside += 1
    return count_inside


def par_pick(n):
    nlocal=int(n/size)
    localcount=pick(nlocal)
    count=localcount
    if rank!=0:
       comm.send(localcount, dest=0)
    if rank==0:
       for irank in range(1,size):
          rcount=comm.recv(source=irank)
          count += rcount
    return count

if (__name__ == "__main__"):
    import sys
    import time
    n = int(sys.argv[1])
    tic = time.time()
    count = par_pick(n)
    pmcpi = 4 * count / n
    toc = time.time()
    if rank == 0:
        print(f'For {n} picks, the approximation of pi is {pmcpi} ({abs(pmcpi-np.pi)/np.pi:.2f} %) compute in {toc - tic:.2f} seconds')
