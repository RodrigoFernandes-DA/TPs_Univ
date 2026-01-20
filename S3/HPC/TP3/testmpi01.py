from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

x = 100 + rank

if rank < size - 1:
    comm.send(x, dest=rank + 1)
    print(f"[P{rank}] Send {x} to P{rank + 1}")

if rank > 0:
    data = comm.recv(source=rank - 1)
    print(f"[P{rank}] Received {data} from P{rank - 1}")

