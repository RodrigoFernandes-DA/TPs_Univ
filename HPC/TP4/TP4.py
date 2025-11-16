from mpi4py import MPI 
import numpy as np
import M2SD_HPC_TP04_MPIConway as conway
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a 2D Cartesian communicator (auto-decompose into near-square grid)
dims = MPI.Compute_dims(size, 2)  # e.g., for 4 ranks -> [2,2]
cart_comm = comm.Create_cart(dims, periods=[False, False], reorder=True)
coords = cart_comm.Get_coords(rank)         # (proc_row, proc_col)
nbr_up, nbr_down = cart_comm.Shift(0, 1)    # vertical neighbors
nbr_left, nbr_right = cart_comm.Shift(1, 1) # horizontal neighbors
nrows_ranks, ncols_ranks = dims

def idim_local(grid):
    nbrows, nbcols = grid.shape   

    # nombre de lignes minimum par processus
    nb_lines_per_proc = nbrows // size       

    # indice de début de mon bloc
    start = rank * nb_lines_per_proc       

    if rank == size - 1:
        # le dernier prend tout ce qu'il reste
        nb_lines_local= nbrows - start
    else:
        nb_lines_local = nb_lines_per_proc

    # on retourne (nb_lignes_locales, indice_depart_global)
    return nb_lines_local, start


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


def create_local_grid(grid, nb_lines_local, start):
    #  nb de lignes + indice de départ global
    # nb_lines_local, start = idim_local(grid)

    # sous-grille locale SANS fantômes (juste les vraies lignes de ce rank)
    local_grid = grid[start:start + nb_lines_local, :]

    #  on élargit cette sous-grille => création automatique
    #    des zones fantômes et interfaces
    local_e_grid = conway.enlarge_grid(local_grid)

    return local_e_grid


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


def conway_p2p(grid, n):
    total_start = MPI.Wtime()
    nbrows, nbcols = grid.shape

    # --- 1) Domaine local + grille locale élargie --------------------------
    nb_lignes_local, start = idim_local(grid)
    local_egrid = create_local_grid(grid, nb_lignes_local, start)
    # local_egrid.shape = (nb_lignes_local + 2, nbcols + 2)
    # lignes 0 et nb_lignes_local+1 = fantômes verticaux

    for step in range(n):

        # --- 2) ÉCHANGES POINT-À-POINT DES INTERFACES ----------------------
        # Interface haut  = ligne 1
        # Interface bas   = ligne nb_lignes_local
        # Fantôme haut    = ligne 0
        # Fantôme bas     = ligne nb_lignes_local + 1

        # voisin du haut
        if rank > 0:
            top = rank - 1
            # On envoie notre interface HAUT (ligne 1)
            # On reçoit dans notre fantôme HAUT (ligne 0)
            comm.Sendrecv(
                sendbuf=local_egrid[1, :],          dest=top,    sendtag=0,
                recvbuf=local_egrid[0, :],          source=top,  recvtag=1
            )

        # voisin du bas
        if rank < size - 1:
            bottom = rank + 1
            # On envoie notre interface BAS (ligne nb_lignes_local)
            # On reçoit dans notre fantôme BAS (ligne nb_lignes_local + 1)
            comm.Sendrecv(
                sendbuf=local_egrid[nb_lignes_local, :],     dest=bottom, sendtag=1,
                recvbuf=local_egrid[nb_lignes_local + 1, :], source=bottom, recvtag=0
            )
            
        # 3) STATISTIQUES (AVANT OU APRÈS life_step, mais toujours comme conway)
        # ⚠ très important : on ne somme que les cellules RÉELLES,
        # pas les fantômes, sinon on double-compte les bords.
        alive_local = np.sum(
            local_egrid[1:nb_lignes_local + 1, 1:nbcols + 1]
        )

        # On additionne tous les alive_local sur le rang 0
        alive_global = comm.reduce(alive_local, op=MPI.SUM, root=0)

        if rank == 0:
            print(f"alive cells: {alive_global}: "
                  f"{alive_global / grid.size * 100:2.2f}%")

        # --- 4) UNE ÉTAPE DU JEU DE LA VIE EN LOCAL ------------------------
        # life_step travaille sur une grille ÉLARGIE (avec fantômes)
        new_local_egrid = conway.life_step(local_egrid)
        local_egrid = new_local_egrid

    # --- 5) EXTRAIRE LA PARTIE RÉELLE (sans fantômes) ----------------------
    # lignes réelles : 1 .. nb_lignes_local
    # colonnes réelles : 1 .. nbcols
    local_result = local_egrid[1:nb_lignes_local + 1, 1:nbcols + 1]

    # --- 6) GATHER : RASSEMBLER LA GRILLE FINALE SUR LE RANG 0 -------------

    # On utilise comm.gather, qui est un "Gather" haut niveau :
    # chaque rank envoie son bloc local_result
    # le rang 0 reçoit une liste de blocs qu'on empile
    all_parts = comm.gather(local_result, root=0)

    if rank == 0:
        # all_parts = [bloc_rank0, bloc_rank1, ..., bloc_rank{size-1}]
        # On empile verticalement selon l'ordre des ranks
        result_grid = np.vstack(all_parts)
        
        total_end = MPI.Wtime()
        return result_grid, total_end - total_start
    else:
        total_end = MPI.Wtime()
        return None, total_end - total_start


def conway_coll(grid, n):
    total_start = MPI.Wtime()
    nbrows, nbcols = grid.shape

    # 1) Domaine local + grille locale élargie
    nb_lignes_local, start = idim_local(grid)
    local_egrid = create_local_grid(grid, nb_lignes_local, start)
    # local_egrid a la forme (nb_lignes_local + 2, nbcols + 2)

    for step in range(n):

        # 2) Préparer les interfaces à envoyer
        # interfaces[0, :] = interface haut  (ligne 1)
        # interfaces[1, :] = interface bas   (ligne nb_lignes_local)
        interfaces = np.empty((2, nbcols + 2), dtype=grid.dtype)
        interfaces[0, :] = local_egrid[1, :]
        interfaces[1, :] = local_egrid[nb_lignes_local, :]

        # 3) Allgather : chaque processus reçoit les interfaces de TOUS les processus
        all_interfaces = comm.allgather(interfaces)
        # all_interfaces[r].shape = (2, nbcols+2) pour chaque rang r

        # 4) Mettre à jour les zones fantômes avec les voisins

        # Voisin du haut : rank - 1 → on prend son interface BAS (index 1)
        if rank > 0:
            haut = rank - 1
            local_egrid[0, :] = all_interfaces[haut][1, :]

        # Voisin du bas : rank + 1 → on prend son interface HAUT (index 0)
        if rank < size - 1:
            bas = rank + 1
            local_egrid[nb_lignes_local + 1, :] = all_interfaces[bas][0, :]

        # 5) STATISTIQUES (AVANT OU APRÈS life_step, mais toujours comme conway)
        # ⚠ très important : on ne somme que les cellules RÉELLES,
        # pas les fantômes, sinon on double-compte les bords.
        alive_local = np.sum(
            local_egrid[1:nb_lignes_local + 1, 1:nbcols + 1]
        )

        # On additionne tous les alive_local sur le rang 0
        alive_global = comm.reduce(alive_local, op=MPI.SUM, root=0)

        if rank == 0:
            print(f"alive cells: {alive_global}: "
                  f"{alive_global / grid.size * 100:2.2f}%")

        # 6) Une étape du jeu de la vie en local
        local_egrid = conway.life_step(local_egrid)

    # 7) Extraire la partie réelle (sans fantômes)
    local_result = local_egrid[1:nb_lignes_local + 1, 1:nbcols + 1]

    # 8) Gather final : reconstituer la grille globale sur le rang 0
    all_parts = comm.gather(local_result, root=0)

    if rank == 0:
        # all_parts = [bloc_rank0, bloc_rank1, ..., bloc_rank{size-1}]
        result_grid = np.vstack(all_parts)
        total_end = MPI.Wtime()
        return result_grid, total_end - total_start
    else:
        total_end = MPI.Wtime()
        return None, total_end - total_start


def conway_block(grid, n):
    total_start = MPI.Wtime()
    (i_start, i_end), (j_start, j_end) = idim_local_blocks(grid.shape, coords)
    
    for ep in range(n):
        local = grid[i_start:i_end, j_start:j_end]
        local_e_grid = create_local_grid_2d(grid, i_start, i_end, j_start, j_end)
        ni, nj = local.shape

        # vertical exchange (up / down)
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

        # horizontal exchange (left / right)
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
        
        # 5) STATISTIQUES (AVANT OU APRÈS life_step, mais toujours comme conway)
        # ⚠ très important : on ne somme que les cellules RÉELLES,
        # pas les fantômes, sinon on double-compte les bords.
        alive_local = np.sum(
            local_e_grid[1:i_end-i_start + 1, 1:j_end-j_start + 1]
        )

        # On additionne tous les alive_local sur le rang 0
        alive_global = comm.reduce(alive_local, op=MPI.SUM, root=0)

        if rank == 0:
            print(f"alive cells: {alive_global}: "
                  f"{alive_global / grid.size * 100:2.2f}%")

        # life_step expects an enlarged grid
        egrid = conway.life_step(local_e_grid)
        local_result = egrid[1:-1, 1:-1]  # remove ghosts

        # Gather pairs of (coords, local_result) on rank 0
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

        # broadcast updated grid for next epoch to all ranks
        grid = comm.bcast(grid if rank == 0 else None, root=0)

    total_end = MPI.Wtime()
    return grid, total_end - total_start
 
    
if __name__ == "__main__":
    grid_sizes = [(10, 10), (100, 100), (1000, 1000), (2000, 2000)]
    # grid_sizes = [(10, 10), (20, 20)]
    epochs = 5
    timings = []
    timings_p2p = []
    timings_coll = []
    timings_block = []

    for gsize in grid_sizes:
        
        # Initialize grid on rank 0
        if rank == 0:
            print(f"\n========= Grid {gsize} =========")
            grid = conway.init_grid(gsize, threshold=0.4)
        else:
            grid = None

        # Broadcast/diffusion to all ranks
        grid = comm.bcast(grid, root=0)
        
        
        # =====================================
        # 2) VERSION SÉQUENTIELLE (référence)
        #    → exécutée uniquement sur le rang 0
        # =====================================
        tic = time.time()
        if rank == 0:
            print("=== Conway normal (séquentiel) ===")
            grid_seq = grid.copy()
            res = conway.conway(grid_seq, epochs)
            
        else:
            res = None  # pour que la variable existe partout
        
        toc = time.time()
        total_time = toc - tic
        timings.append((gsize[0], gsize[1], "Seq", total_time))
        
        # Synchronisation avant la version MPI
        comm.Barrier()
        
        
        # =====================================
        # 3) VERSION MPI POINT-TO-POINT (conway_p2p)
        # =====================================
        if rank == 0:
            print("\n=== Conway P2P (MPI point-à-point) ===")
        grid_p2p = grid.copy()
        res_p2p, exec_time_p2p = conway_p2p(grid_p2p, epochs)
        
        # Collect times from all ranks and take the max (slowest rank determines runtime)
        total_time = comm.reduce(exec_time_p2p, op=MPI.MAX, root=0)
        
        if rank == 0:
            timings_p2p.append((gsize[0], gsize[1], "P2P", total_time))

        # Synchronisation avant la version collective
        comm.Barrier()
        
        
        # =====================================
        # 4) VERSION MPI COLLECTIVE (conway_coll)
        # =====================================
        if rank == 0:
            print("\n=== Conway collective (MPI allgather) ===")
        grid_coll = grid.copy()
        res_coll, exec_time_coll = conway_coll(grid_coll, epochs)
        
        # Collect times from all ranks and take the max (slowest rank determines runtime)
        total_time = comm.reduce(exec_time_coll, op=MPI.MAX, root=0)
        
        if rank == 0:
            timings_coll.append((gsize[0], gsize[1], "Coll", total_time))
        
        # Synchronisation 
        comm.Barrier()
        
        
        # =====================================
        # 5) VERSION BLOCK (conway_block)
        # =====================================
        if rank == 0:
            print("\n=== Conway Block (2d) ===")
        grid_block = grid.copy()
        res_block, exec_time_block = conway_block(grid_block, epochs)
        
        # Collect times from all ranks and take the max (slowest rank determines runtime)
        total_time = comm.reduce(exec_time_block, op=MPI.MAX, root=0)
        
        if rank == 0:
            timings_block.append((gsize[0], gsize[1], "Block", total_time))
        
        # Synchronisation 
        comm.Barrier()


    # =====================================
    # 6) VÉRIFICATION DES RÉSULTATS
    # =====================================
    if rank == 0:
        print("\n=== Vérification des résultats finaux ===")
        print("OK P2P  ? ", np.array_equal(res, res_p2p))
        print("OK Coll ? ", np.array_equal(res, res_coll))
        print("OK Block ? ", np.array_equal(res, res_block))
        
        
    # =====================================
    # 7) Print result
    # =====================================
    
        print("\n=== Tableau récapitulatif des temps d'exécution ===")

        all_timings = timings + timings_p2p + timings_coll + timings_block

        print(f"{'Grid Size':>12} {'GridY':>8} {'Method':>10} {'Time (s)':>12}")
        print("-" * 42)
        for gx, gy, method, t in all_timings:
            print(f"{str((gx, gy)):>12} {method:10s} {t:12.6f}")

    
    