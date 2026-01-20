# conway_coll.py  -- 2D Cartesian decomposition (fixed reconstruction bug)
from mpi4py import MPI
import numpy as np
import M2SD_HPC_TP04_MPIConway as conway

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a 2D Cartesian communicator (auto-decompose into near-square grid)
dims = MPI.Compute_dims(size, 2)  # e.g., for 4 ranks -> [2,2]
cart_comm = comm.Create_cart(dims, periods=[False, False], reorder=False)
coords = cart_comm.Get_coords(rank)         # (proc_row, proc_col)
nbr_up, nbr_down = cart_comm.Shift(0, 1)    # vertical neighbors
nbr_left, nbr_right = cart_comm.Shift(1, 1) # horizontal neighbors
nrows_ranks, ncols_ranks = dims


def idim_local_blocks(grid_shape, proc_coords):
    g_rows, g_cols = grid_shape
    pr, pc = proc_coords

    base_r = g_rows // nrows_ranks
    rem_r = g_rows % nrows_ranks
    i_start = pr * base_r + min(pr, rem_r)
    i_rows = base_r + (1 if pr < rem_r else 0)
    i_end = i_start + i_rows

    base_c = g_cols // ncols_ranks
    rem_c = g_cols % ncols_ranks
    j_start = pc * base_c + min(pc, rem_c)
    j_cols = base_c + (1 if pc < rem_c else 0)
    j_end = j_start + j_cols

    return (i_start, i_end), (j_start, j_end)



def create_local_grid_2d(grid, i_start, i_end, j_start, j_end):
    # slice using end as exclusive
    if i_end <= i_start or j_end <= j_start:
        # this rank got zero rows or zero cols
        nrows_local = max(0, i_end - i_start)
        ncols_local = max(0, j_end - j_start)
        return np.zeros((2, ncols_local + 2), dtype=grid.dtype), (nrows_local, ncols_local)

    local = grid[i_start:i_end, j_start:j_end]
    local_e_grid = conway.enlarge_grid(local)
    return local_e_grid



def conway_block(grid, n):
    total_start = MPI.Wtime()
    
    (i_start, i_end), (j_start, j_end) = idim_local_blocks(grid.shape, coords)
    
    for ep in range(n):
        
        # 1) Domaine local + grille locale élargie
        local = grid[i_start:i_end, j_start:j_end]
        local_e_grid = create_local_grid_2d(grid, i_start, i_end, j_start, j_end)
        ni, nj = local.shape

        # 2) vertical exchange (up / down)
        if nbr_up != MPI.PROC_NULL:
            send_up = local[0, :].copy()
            recv_up = np.empty_like(send_up)
            cart_comm.Sendrecv(send_up, dest=nbr_up, recvbuf=recv_up, source=nbr_up)
        else:
            recv_up = np.zeros(nj, dtype=local.dtype)

        if nbr_down != MPI.PROC_NULL:
            send_down = local[-1, :].copy()
            recv_down = np.empty_like(send_down)
            cart_comm.Sendrecv(send_down, dest=nbr_down, recvbuf=recv_down, source=nbr_down)
        else:
            recv_down = np.zeros(nj, dtype=local.dtype)

        local_e_grid[0, 1:-1] = recv_up
        local_e_grid[-1, 1:-1] = recv_down

        comm.Barrier()

        # 3) horizontal exchange (left / right)
        if nbr_left != MPI.PROC_NULL:
            send_left = local[:, 0].copy()
            recv_left = np.empty_like(send_left)
            cart_comm.Sendrecv(send_left, dest=nbr_left, recvbuf=recv_left, source=nbr_left)
        else:
            recv_left = np.zeros(ni, dtype=local.dtype)
            

        if nbr_right != MPI.PROC_NULL:
            send_right = local[:, -1].copy()
            recv_right = np.empty_like(send_right)
            cart_comm.Sendrecv(send_right, dest=nbr_right, recvbuf=recv_right, source=nbr_right)
        else:
            recv_right = np.zeros(ni, dtype=local.dtype)

        local_e_grid[1:-1, 0] = recv_left
        local_e_grid[1:-1, -1] = recv_right
        
        # 4) STATISTIQUES 
        alive_local = np.sum(
            local_e_grid[1:i_end-i_start + 1, 1:j_end-j_start + 1]
        )

        # On additionne tous les alive_local sur le rang 0
        alive_global = comm.reduce(alive_local, op=MPI.SUM, root=0)

        if rank == 0:
            print(f"alive cells: {alive_global}: "
                  f"{alive_global / grid.size * 100:2.2f}%")

        # 5) Une étape du jeu de la vie en local
        egrid = conway.life_step(local_e_grid)
        local_result = egrid[1:-1, 1:-1]  # remove ghosts

        # 6) Gather pairs of (coords, local_result) on rank 0
        gathered = comm.gather((coords, local_result), root=0)

        if rank == 0:
            # Reconstruct full grid from gathered tiles using their reported coords
            full = np.zeros_like(grid)
            for proc_coords, sub in gathered:
                (i_s, i_e), (j_s, j_e) = idim_local_blocks(grid.shape, proc_coords)

                # sub can be empty if that tile has zero size; skip then
                if sub.size == 0:
                    continue

                # sanity check: shapes must match the computed tile dims
                expected_shape = (i_e - i_s, j_e - j_s)
                if sub.shape != expected_shape:
                    raise ValueError(f"Subtile shape mismatch for proc {proc_coords}: "
                                     f"got {sub.shape}, expected {expected_shape}")

                full[i_s:i_e, j_s:j_e] = sub
            grid = full
            
        comm.Barrier()

        # broadcast updated grid for next epoch to all ranks
        grid = comm.bcast(grid if rank == 0 else None, root=0)

    total_end = MPI.Wtime()
    return grid, total_end - total_start
 


# driver
if __name__ == "__main__":
    grid_sizes = [(10, 10), (100, 100), (200, 200)]
    epochs = 2
    timings = []

    for gsize in grid_sizes:
        if rank == 0:
            grid = conway.init_grid(gsize, threshold=0.4)
        else:
            grid = None


        if rank == 0:
            print("=== Conway normal (séquentiel) ===")
            grid_seq = grid.copy()
            res = conway.conway(grid_seq, epochs)
            
        else:
            res = None  # pour que la variable existe partout

        comm.Barrier()
        # DON'T BROADCAST THE GRID
        grid = comm.bcast(grid, root=0)

        res_block, exec_time = conway_block(grid, epochs)

        total_time = comm.reduce(exec_time, op=MPI.MAX, root=0)
        
        comm.Barrier()

        if rank == 0:
            timings.append((gsize[0], gsize[1], epochs, total_time))
            print("Size:", gsize, "Time:", total_time)
            
        if rank == 0:
            print("\n=== Vérification des résultats finaux ===")
            print("OK Block ? ", np.array_equal(res, res_block))

    # if rank == 0:
    #     print("\n=== 2D Domain Decomposition Performance (fixed) ===")
    #     print(f"{'Grid Size':>12} | {'Epochs':>6} | {'Exec Time (s)':>20}")
    #     print("-" * 45)
    #     for n, m, ep, t in timings:
    #         print(f"{str((n,m)):>12} | {ep:>6} | {t:>20.6f}")
    #     print("-" * 45)
