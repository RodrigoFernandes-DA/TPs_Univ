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
    """Calcul parallèle du nombre de points dans le cercle avec réduction MPI."""
    nlocal = int(n / size)
    localcount = pick(nlocal)

    # Communication collective : réduction (somme des résultats vers le processus 0)
    total_count = comm.reduce(localcount, op=MPI.SUM, root=0)

    return total_count

if __name__ == "__main__":
    n = int(sys.argv[1])
    tic = time.time()
    count = par_pick(n)

    if rank == 0:  # Seul le processus 0 détient le résultat final
        pmcpi = 4 * count / n
        toc = time.time()
        print(f"For {n} picks, the approximation of pi is {pmcpi:.6f} "
              f"({abs(pmcpi - np.pi)/np.pi:.2%} error), computed in {toc - tic:.2f} s")
