import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector

def gen_occupany(image):
    # Load the PNG image
    # image_path = "images/grid.png"
    # image = cv2.imread(image_path)
    # if image is None:
        # raise FileNotFoundError(f"Could not load image at {image_path}")

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
        return np.zeros((100, 100, 3), dtype=np.uint8)

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
    grid_size = (100, 100)
    height, width = image.shape[:2]
    grid = cv2.resize(cleaned_mask, grid_size, interpolation=cv2.INTER_NEAREST)
    grid = (grid == 0).astype(np.uint8)  # 0 = carpet (free), 1 = obstacle

    # Initial connected components filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    min_obstacle_size = max(20, (grid_size[0] * grid_size[1]) // 500)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_obstacle_size:
            grid[labels == i] = 0

    # Scale robot and destination positions to grid coordinates
    robot_grid_x = int(robot_pos[0] * grid_size[1] / width)
    robot_grid_y = int(robot_pos[1] * grid_size[0] / height)
    dest_grid_x = int(dest_pos[0] * grid_size[1] / width)
    dest_grid_y = int(dest_pos[1] * grid_size[0] / height)

    # Clear white borders around AprilTags
    tag_radius = 10
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            if ((x - robot_grid_x) ** 2 + (y - robot_grid_y) ** 2) <= tag_radius ** 2:
                grid[y, x] = 0
            if ((x - dest_grid_x) ** 2 + (y - dest_grid_y) ** 2) <= tag_radius ** 2:
                grid[y, x] = 0

    # Mark positions on the grid
    grid[robot_grid_y, robot_grid_x] = 2  # Robot = 2
    grid[dest_grid_y, dest_grid_x] = 3     # Destination = 3

    # Final noise reduction: remove tiny obstacles
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    final_min_size = 10  # Smaller threshold for final cleanup, adjust as needed
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < final_min_size and grid[labels == i].max() < 2:
            grid[labels == i] = 0  # Only clear if no robot/destination (2 or 3) in the component

    # Print the grid
    # print("Occupancy Grid (0=free carpet, 1=obstacle, 2=robot, 3=destination):")
    # print(grid)


    # Create visual grid
    grid_visual = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.uint8)
    grid_visual[grid == 0] = [255, 255, 255]  # White = free (carpet)
    grid_visual[grid == 1] = [0, 0, 0]        # Black = obstacle
    grid_visual[grid == 2] = [0, 255, 0]      # Green = robot
    grid_visual[grid == 3] = [0, 0, 255]      # Blue = destination

    # Resize images for display
    original_display = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    grid_display = cv2.resize(grid_visual, (500, 500), interpolation=cv2.INTER_NEAREST)

    # Display both images
    # cv2.imshow("Original Image", original_display)
    # cv2.imshow("Occupancy Grid", grid_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Save the grid
    # np.save("occupancy_grid.npy", grid)
    if grid is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    return grid_display