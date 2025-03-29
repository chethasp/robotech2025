import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector
from collections import deque

# Constants
GRID_SIZE = 100
ROBOT_RADIUS = 2  # 5x5 robot in 100x100 grid
SAFE_DISTANCE = 10  # Buffer zone size
MIN_CLUSTER_SIZE = 25  # Minimum size for obstacle clusters

def create_path(image_path):
    # Load the PNG image
    # image_path = "images/grid.png"
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

    # Initial Noise Reduction
    blurred_mask = cv2.GaussianBlur(carpet_mask, (5, 5), 0)
    kernel = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Resize to occupancy grid
    height, width = image.shape[:2]
    grid = cv2.resize(cleaned_mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
    grid = (grid == 0).astype(np.uint8)  # 0 = carpet (free), 1 = obstacle

    # Noise filter function
    def filter_small_noise(grid, min_size):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                grid[labels == i] = 0
        return grid

    # First noise filter
    grid = filter_small_noise(grid, MIN_CLUSTER_SIZE)

    # Additional smoothing
    kernel = np.ones((3, 3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

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

    # Second noise filter
    grid = filter_small_noise(grid, MIN_CLUSTER_SIZE)

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

    # BFS pathfinding algorithm
    def bfs_pathfinder(grid, start, goal):
        queue = deque([start])
        visited = set([start])
        parent = {start: None}  # Track parent nodes to reconstruct path
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, down, left, right
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
        
        while queue:
            current_x, current_y = queue.popleft()
            
            if (current_x, current_y) == goal:
                # Reconstruct path
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]  # Reverse to get start-to-goal order
            
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                    grid[ny, nx] not in [1, 4] and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (current_x, current_y)
        
        print("No valid path found with BFS")
        return None

    # Find path
    start = (robot_grid_x, robot_grid_y)
    goal = (dest_grid_x, dest_grid_y)
    path = bfs_pathfinder(grid, start, goal)

    # Print results
    if path:
        print(f"Path found with {len(path)} steps")
        print(f"Robot: {start}, Target: {goal}")
        print(f"Path: {path}")
    else:
        print("No path found!")

    # Visualize the grid and path
    grid_visual = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    grid_visual[grid == 0] = [255, 255, 255]  # White = free (carpet)
    grid_visual[grid == 1] = [0, 0, 0]        # Black = obstacle
    grid_visual[grid == 4] = [64, 64, 64]     # Gray = buffer zone
    grid_visual[grid == 2] = [0, 0, 255]      # Blue = robot
    grid_visual[grid == 3] = [0, 255, 0]      # Green = destination

    # Draw the path if it exists
    if path is not None and len(path) > 0:
        for x, y in path:
            if grid[y, x] not in [2, 3]:  # Don’t overwrite robot or destination
                grid_visual[y, x] = [255, 255, 0]  # Yellow = path
    else:
        print("Path is None or empty, skipping visualization")

    # Resize images for display
    original_display = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    grid_display = cv2.resize(grid_visual, (500, 500), interpolation=cv2.INTER_NEAREST)

    # Display both images
    cv2.imshow("Original Image", original_display)
    cv2.imshow("Occupancy Grid with Path", grid_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the grid
    np.save("occupancygrids/occupancy_grid_with_path.npy", grid)

    return path

# create_path("images/grid.png")