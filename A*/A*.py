from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
import matplotlib.pyplot as plt


def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h : float = 0.0, parent: Dict = None) -> Dict : 
    '''
    1. position : Tuple[int, int] : Posicion del nodo en el mapa
    
    2. g : flaot = float ("inf") : Costo Real desde el nodo de inicio hasta nodo
                # Se inicializa en infinito por defecto del A*
    
    3. h : float = 0.0 : heuristica estimada

    4. Dict = None : Nodo Padre desde el cual se llego al siguiente nodo. 
        Para reconstruir camino finla

    5. -> : Regresa un nodo del Dictionario
                
    '''
    return {
        'position': position ,
        'g' : g, 
        'h' : h,
        'f' : g + h, 
        'parent' : parent
    }



def heuristic (pos1 : Tuple [int, int], pos2 : Tuple[int, int]) -> float : 
    # Calculate Heuristic

    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def valid_neighbors(grid : np.ndarray, position: Tuple[int, int]) ->List[Tuple[int, int]]: 
    '''
    
    Get all valid neighbors positions in the grid/map

    1. Grid : 2D np array where 0 is walkable cells and 1 is objects
    
    2. postion : Current position (x,y)

    Returns : List valid neighboring positions
    
    '''

    x, y = position
    rows, colms = grid.shape 

    # Calc Pos moves (diagonals include)
    poss_moves = [
        (x+1, y), (x-1, y), # Moving Right, left
        (x, y+1), (x, y-1), # Moving Up, Down
        (x+1, y+1), (x-1, y-1), # moving diagonal
        (x+1, y-1), (x-1, y+1)
    ]

    return [ 
        (new_x, new_y) for new_x, new_y in poss_moves 
        if 0 <= new_x < rows and 0 <= new_y < colms  # New Point within the boundaries of map
        and grid [new_x, new_y] == 0 # Cell is a path and not an Object (1)
    ]

def reconstruct_path (goal_node: Dict)-> List[Tuple[int, int]] : 
    # Reconstruct Path from goal to start by following parents

    path = []
    current = goal_node 

    while current is not None : 
        path.append(current['position'])
        current = current['parent']

    return path [::-1] # TO ger path from start to goal

def find_path (grid : np.ndarray, start: Tuple[int, int],
               goal: Tuple[int, int]) -> List[Tuple[int, int]]: 
    
    '''
    Find optimal Path 

    1. Grid : 2D numpy array (0 = Free space , 1 = obstacle)
    2. Start : (x,y)
    3. Goal : (x, y)

    Returns : List positions representing optimal path

    '''
    start_node = create_node(
        position = start, 
        g = 0 , 
        h = heuristic(start, goal)
    )

    # Sets
    open_list = [(start_node['f'], start)] # Priority Queue
    open_dict = {start: start_node}  # Posicion del nodo en el mapa
    closed_set = set() # Pos ya evaluado

    while open_list :  # Mientras haya nodo por explorar
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list) 
        current_node = open_dict[current_pos] # extrae nodo con menor costo

        # Check if we've reached goal
        if current_pos == goal :  
            return reconstruct_path(current_node)
        
        # Check neighbors 
        for neighbor_pos in valid_neighbors(grid, current_pos): 
            if neighbor_pos in closed_set :  # ignore already explored neighbors
                continue

            # Calc new path cost
            tentative_g = current_node['g'] + heuristic(current_pos, neighbor_pos) # Costo total desde el inicio hasta nuevo vencino
            # Create or update neighbor
            if neighbor_pos not in open_dict :  # Si la primera vez que se visita un nodo, 
                neighbor = create_node(
                    position= neighbor_pos, 
                    g = tentative_g , 
                    h = heuristic (neighbor_pos, goal), 
                    parent = current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict [neighbor_pos]['g']: 
                # Found better path 
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node

    return [] # no path found 

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]):
    """
    Visualize the grid and found path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')
    
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("A* Pathfinding Result")
    plt.savefig("A*.png")
    #plt.show()

def create_grid() -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Crea un grid con obstáculos y devuelve grid, posición inicial y final.
    """
    grid = np.zeros((30, 50))
    
    # Obstáculos
    grid[5:15, 10] = 1
    grid[5, 5:15] = 1
    grid[2, 2:10] = 1
    grid[10, 15:35] = 1

    start_pos = (2, 2)
    goal_pos = (29, 47)

    return grid, start_pos, goal_pos

grid, start_pos, goal_pos = create_grid()
path = find_path(grid, start_pos, goal_pos)
if path : 
    print(f'path found with {len(path)} steps')
    visualize_path(grid, path)
else : 
    print('no path')