from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def pick(n):
    """Renvoie le nombre de points dans le cercle unité sur n tirages aléatoires"""
    count_inside = 0
    for i in range(n):
        x, y = np.random.random(2) * 2 - 1  # Tirage dans [-1,1] × [-1,1]
        if x**2 + y**2 <= 1:
            count_inside += 1
    return count_inside

def par_pick(n):
    """Calcul parallèle du nombre de points dans le cercle avec communication collective (allreduce)."""
    nlocal = int(n / size)
    localcount = pick(nlocal)

    # Communication collective : somme des résultats partiels sur TOUS les processus
    total_count = comm.allreduce(localcount, op=MPI.SUM)

    return total_count

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
