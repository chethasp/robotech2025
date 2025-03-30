import cv2
import numpy as np
from collections import deque
from heapq import heappop, heappush

# Constants
GRID_SIZE = 100
MAP_SIZE_METERS = 1.502475
CELL_SIZE_METERS = MAP_SIZE_METERS / GRID_SIZE
ROBOT_RADIUS = 2  # 5x5 robot in 100x100 grid
PATH_THICKNESS = 2
SAFE_DISTANCE = 6  # Buffer zone size

def create_occupancy_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    # Smaller central obstacle (10x10 instead of 16x16)
    grid[45:55, 45:55] = 1  
    # Horizontal wall (5x15)
    grid[20:25, 10:25] = 1  
    # Small triangle (5 tall)
    for i in range(65, 70):    
        width = int((70 - i) * 1.0)
        grid[i, 77 - width:77 + width] = 1
    return grid

def add_safety_buffer(grid, safe_distance):
    """Create buffer zone around obstacles"""
    obstacle_map = (grid == 1).astype(np.uint8) * 255
    dist_map = cv2.distanceTransform(~obstacle_map, cv2.DIST_L2, 3)
    buffer_zone = (dist_map <= safe_distance).astype(np.uint8)
    grid[(buffer_zone == 1) & (grid != 1)] = 4
    return grid

def world_to_grid(x, y):
    grid_x = int(np.clip(x / CELL_SIZE_METERS, 0, GRID_SIZE - 1))
    grid_y = int(np.clip(y / CELL_SIZE_METERS, 0, GRID_SIZE - 1))
    return grid_x, grid_y

def update_occupancy_grid(grid, robot_pos, target_pos):
    # Update robot position (blue)
    rx, ry = robot_pos
    half = ROBOT_RADIUS
    grid[max(0, ry-half):min(GRID_SIZE, ry+half+1),
         max(0, rx-half):min(GRID_SIZE, rx+half+1)] = 2
    
    # Update target position (green, top-right)
    tx, ty = target_pos
    grid[max(0, ty-half):min(GRID_SIZE, ty+half+1),
         max(0, tx-half):min(GRID_SIZE, tx+half+1)] = 3
    return grid

def bfs_pathfinder(grid, start, goal):
    # 8-direction movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal (cost 1)
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal (cost sqrt(2))
    
    queue = []
    heappush(queue, (0, start))  # (cost, position)
    
    visited = np.full((GRID_SIZE, GRID_SIZE), False)
    parent = {}
    cost_so_far = {start: 0}
    
    visited[start[1], start[0]] = True
    parent[start] = None

    while queue:
        current_cost, current = heappop(queue)
        
        if current == goal:
            break
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            new_cost = current_cost + (1.414 if dx != 0 and dy != 0 else 1)
            
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                not visited[ny, nx] and
                grid[ny, nx] not in [1, 4]):  # Avoid obstacles & buffer
                
                # Prevent cutting through obstacles diagonally
                if dx != 0 and dy != 0:
                    if grid[current[1], nx] in [1, 4] or grid[ny, current[0]] in [1, 4]:
                        continue

                # If new path is better, update cost and push to queue
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    heappush(queue, (new_cost, (nx, ny)))
                    parent[(nx, ny)] = current
                    visited[ny, nx] = True

    # Reconstruct path
    if goal not in parent:
        return None
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    
    return path[::-1]  # Reverse to get start to goal

def visualize(grid, path=None):
    display = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    display[grid == 0] = [0, 0, 0]       # Free space
    display[grid == 1] = [128, 128, 128] # Obstacles
    display[grid == 4] = [64, 64, 64]    # Buffer
    display[grid == 2] = [0, 0, 255]     # Robot (blue)
    display[grid == 3] = [0, 255, 0]     # Target (green)
    
    if path:
        for x, y in path:
            if grid[y, x] not in [2, 3]:
                display[y, x] = [255, 255, 0]  # Path (yellow)
    
    # Scale up for visibility
    return cv2.resize(display, (500, 500), interpolation=cv2.INTER_NEAREST)

def main():
    # Setup grid with obstacles
    grid = create_occupancy_grid()
    grid = add_safety_buffer(grid, SAFE_DISTANCE)
    
    # Robot position (bottom-left)
    robot_pos = world_to_grid(0.3, 1.2)
    
    # Target position (top-right corner, 90% of map size)
    target_pos = (int(GRID_SIZE*0.9), int(GRID_SIZE*0.1))
    
    # Update grid
    grid = update_occupancy_grid(grid, robot_pos, target_pos)
    
    # Find path using BFS
    path = bfs_pathfinder(grid, robot_pos, target_pos)
    
    # Results
    if path:
        print(f"Path found with {len(path)} steps")
        print(f"Robot: {robot_pos}, Target: {target_pos}")
    else:
        print("No path found!")
    
    # Show visualization
    cv2.imshow('Path Planning', visualize(grid, path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()