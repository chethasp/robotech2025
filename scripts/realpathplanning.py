import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector
from collections import deque

# Constants
GRID_SIZE = 100
ROBOT_RADIUS = 2  # 5x5 robot in 100x100 grid
SAFE_DISTANCE = 4  # Buffer zone size
MIN_OBSTACLE_SIZE = 5  # Minimum size for an obstacle to be considered

# Load the PNG image
image_path = "images/grid.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Convert to grayscale for AprilTag detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize AprilTag detector (assuming 36h11 family)
detector = AprilTagDetector()
detector.addFamily("tag36h11")

# Detect AprilTags
detections = detector.detect(gray)

# Extract robot and destination positions
robot_pos = None
dest_pos = None
for detection in detections:
    center = detection.getCenter()
    x, y = int(center.x), int(center.y)
    if detection.getId() == 4:  # Robot position
        robot_pos = (x, y)
    elif detection.getId() == 2:  # Destination
        dest_pos = (x, y)

if robot_pos is None or dest_pos is None:
    raise ValueError("Could not find both AprilTags (ID 4 and ID 2)")

print(f"Robot position: {robot_pos}")
print(f"Destination position: {dest_pos}")

# Convert image to HSV for carpet color detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define carpet color range in HSV (your grayish carpet)
lower_carpet = np.array([5, 0, 18])
upper_carpet = np.array([45, 79, 118])
carpet_mask = cv2.inRange(hsv, lower_carpet, upper_carpet)

# Noise Reduction Steps
blurred_mask = cv2.GaussianBlur(carpet_mask, (5, 5), 0)
kernel = np.ones((7, 7), np.uint8)
cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

# Resize to occupancy grid
height, width = image.shape[:2]
grid = cv2.resize(cleaned_mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
grid = (grid == 0).astype(np.uint8)  # 0 = carpet (free), 1 = obstacle

# BFS to filter small obstacles
def bfs_obstacle_filter(grid, min_size):
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-direction movement
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y, x] == 1 and not visited[y, x]:
                queue = deque([(x, y)])
                visited[y, x] = True
                component = [(x, y)]
                
                while queue and len(component) <= min_size:
                    cx, cy = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                            grid[ny, nx] == 1 and not visited[ny, nx]):
                            queue.append((nx, ny))
                            visited[ny, nx] = True
                            component.append((nx, ny))
                
                # If component size is less than min_size, mark as free space
                if len(component) < min_size:
                    for cx, cy in component:
                        grid[cy, cx] = 0

    return grid

# Apply BFS filter
grid = bfs_obstacle_filter(grid, MIN_OBSTACLE_SIZE)

# Scale robot and destination positions to grid coordinates
robot_grid_x = int(robot_pos[0] * GRID_SIZE / width)
robot_grid_y = int(robot_pos[1] * GRID_SIZE / height)
dest_grid_x = int(dest_pos[0] * GRID_SIZE / width)
dest_grid_y = int(dest_pos[1] * GRID_SIZE / height)

# Clear white borders around AprilTags
tag_radius = 10
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        if ((x - robot_grid_x) ** 2 + (y - robot_grid_y) ** 2) <= tag_radius ** 2:
            grid[y, x] = 0
        if ((x - dest_grid_x) ** 2 + (y - dest_grid_y) ** 2) <= tag_radius ** 2:
            grid[y, x] = 0

# Add safety buffer around obstacles
def add_safety_buffer(grid, safe_distance):
    obstacle_map = (grid == 1).astype(np.uint8) * 255
    dist_map = cv2.distanceTransform(~obstacle_map, cv2.DIST_L2, 3)
    buffer_zone = (dist_map <= safe_distance).astype(np.uint8)
    grid[(buffer_zone == 1) & (grid != 1)] = 4  # Buffer zone marked as 4
    return grid

grid = add_safety_buffer(grid, SAFE_DISTANCE)

# Update grid with robot and destination
half = ROBOT_RADIUS
grid[max(0, robot_grid_y-half):min(GRID_SIZE, robot_grid_y+half+1),
     max(0, robot_grid_x-half):min(GRID_SIZE, robot_grid_x+half+1)] = 2  # Robot = 2
grid[max(0, dest_grid_y-half):min(GRID_SIZE, dest_grid_y+half+1),
     max(0, dest_grid_x-half):min(GRID_SIZE, dest_grid_x+half+1)] = 3  # Destination = 3

# Greedy pathfinding algorithm
def greedy_pathfinder(grid, start, goal):
    path = [start]
    current = start
    
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),          (0,1),
                  (1,-1),  (1,0),  (1,1)]
    
    while current != goal:
        best_dist = float('inf')
        best_move = None
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                grid[ny, nx] not in [1, 4]):  # Avoid obstacles and buffer
                
                dist = (goal[0]-nx)**2 + (goal[1]-ny)**2
                if dist < best_dist:
                    best_dist = dist
                    best_move = (nx, ny)
        
        if best_move is None:
            return None  # Stuck
        
        current = best_move
        path.append(current)
        
        if len(path) > GRID_SIZE * 2:
            return None
    
    return path

# Find path
start = (robot_grid_x, robot_grid_y)
goal = (dest_grid_x, dest_grid_y)
path = greedy_pathfinder(grid, start, goal)

# Print results
if path:
    print(f"Path found with {len(path)} steps")
    print(f"Robot: {start}, Target: {goal}")
else:
    print("No path found!")

# Visualize the grid and path
grid_visual = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
grid_visual[grid == 0] = [255, 255, 255]  # White = free (carpet)
grid_visual[grid == 1] = [0, 0, 0]        # Black = obstacle
grid_visual[grid == 4] = [64, 64, 64]     # Gray = buffer zone
grid_visual[grid == 2] = [0, 0, 255]      # Blue = robot
grid_visual[grid == 3] = [0, 255, 0]      # Green = destination

if path:
    for x, y in path:
        if grid[y, x] not in [2, 3]:
            grid_visual[y, x] = [255, 255, 0]  # Yellow = path

# Resize images for display
original_display = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
grid_display = cv2.resize(grid_visual, (500, 500), interpolation=cv2.INTER_NEAREST)

# Display both images
cv2.imshow("Original Image", original_display)
cv2.imshow("Occupancy Grid with Path", grid_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grid
np.save("occupancy_grid_with_path.npy", grid)