import numpy as np
ctype = 'float64'

def init_grid(shape, threshold=0.50):
    """
    initialize a random  grid of ctype 
    of dimension shape (the threshold help to define density)
    """
    x = np.zeros(shape,dtype=ctype)
    r = np.random.random(shape)
    x[:,:] = (r> 1.0 - threshold)
    return x

def read_grid(file):
    """
    load a grid from a ndarray binary file 
    """
    with open(file,'rb') as f:
        grid = np.load(f)
    return grid

def enlarge_grid(grid):
    """
    input: a standard grid as a numpy array
    output: an enlarged grid for edge management
    """
    idim,jdim = grid.shape
    res_grid = np.zeros((idim + 2,jdim + 2),dtype=ctype)
    res_grid[1:-1,1:-1] = grid[:,:]
    return res_grid

def life_step(grid):
    """
    input: enlarged grid
    output: enlarged grid resulting of applying Conway GoL rules
    """
    idim,jdim= grid.shape
    res_grid = np.zeros((idim,jdim), dtype=ctype)
    for i in range(1,idim-1):
        #specific case j=1 for initial storage values
        store0 = 0 # because store0=grid[i-1,0]+grid[i,0]+grid[i+1,0] all zeros
        store1 = grid[i - 1,1] + grid[i,1] + grid[i + 1,1] # j=1
        store2 = grid[i - 1,2] + grid[i,2] + grid[i + 1,2] # j+1=2
        nb_neigh = 0 + store0 + store1 + store2 - grid[i,1] # substract the element (i,j)
        if nb_neigh == 2:
            res_grid[i,1] = grid[i,1]
        elif nb_neigh == 3:
            res_grid[i,1] = 1
        # loop on j (we begin with j=2)
        for j in range(2,jdim - 1):
            # switch the storage values
            store0 = store1
            store1 = store2
            #compute the 3rd storage value
            store2 = grid[i - 1,j + 1] + grid[i,j + 1] + grid[i + 1,j + 1]   # 3rd subcolumn
            # compute nb_neigh
            nb_neigh = store0 + store1 + store2 - grid[i,j] # substract the element (i,j)
            if nb_neigh == 2:
                res_grid[i,j] = grid[i,j]
            elif nb_neigh == 3:
                res_grid[i,j] = 1
    return res_grid

def conway(grid, n):
    """
    From an initial grid, run n iterations of evolution and display 
    the intermediate statistics (number of alive cells and percents)
    """
    # Enlarge the grid
    egrid = enlarge_grid(grid)
    for i in range(n):
        # statstistic on the grid
        alive = np.sum(egrid)
        print(f'alive cells: {alive}: {alive / grid.size * 100:2.2f}%')
        # Apply life_step
        egrid = life_step(egrid)
    # return the standard grid without     
    return egrid[1:-1,1:-1]


def  idim_local(grid, size):
    "en fonction du rang du processus retourne le nombre de ligne locale à traiter ainsi que la position (coordonnée de la 1ère ligne) de la sous-grille dans la grille globale"
    
    
    for rank in range(size):
        irange, jrange = grid.shape
        
        x = np.floor(irange/size)
        
        init = x * rank
        fin = init + x -1
        
        if rank == size-1:
            fin = irange-1
        
        print(f"For rank = {rank}")
        print(f"init = {init}, \nfin = {fin}, \n n = {fin-init+1}")
    
    return np.int(init), np.int(fin)


if __name__ == '__main__':
    grid = read_grid('jdv_1M.grid')
    # res = conway(grid, 5)
    
    idim_local(grid, size = 6)