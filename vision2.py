import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector

# Constants for the grid
INCH_TO_METERS = 0.0254  # Conversion factor: 1 inch = 0.0254 meters
CELL_SIZE_METERS = 0.5 * INCH_TO_METERS  # Each cell represents half an inch (in meters)

def create_occupancy_grid(map_size, cell_size):
    # Define the grid dimensions based on map size and cell size
    grid_width = int(np.ceil(map_size / cell_size))
    grid_height = int(np.ceil(map_size / cell_size))
    # Initialize the grid with zeros (free cells)
    return np.zeros((grid_height, grid_width), dtype=np.uint8)

def world_to_grid(x, y, cell_size):
    # Convert world coordinates (meters) to grid coordinates (cell indices)
    grid_x = int(np.floor(x / cell_size))
    grid_y = int(np.floor(y / cell_size))
    return grid_x, grid_y

def update_occupancy_grid(grid, car_x, car_y, cell_size):
    # Convert car position from meters to grid coordinates
    grid_x, grid_y = world_to_grid(car_x, car_y, cell_size)
    # Check if the coordinates are within the grid bounds
    if 0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]:
        grid[grid_y, grid_x] = 1  # Mark the cell as occupied (car's position)
    return grid

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to handle varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def main():
    # Load the JPG image
    image_path = "fullmap.jpg"  # Replace with your image path
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from '{image_path}'.")
        return

    # Preprocess the image for better detection under varying lighting
    processed_frame = preprocess_image(frame)
    # Create a color version of the processed frame for visualization (since binary is grayscale)
    processed_frame_display = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Create an AprilTag detector and add the tag36h11 family
    detector = AprilTagDetector()
    detector.addFamily("tag36h11")

    # Detect AprilTags in the preprocessed frame
    detections = detector.detect(processed_frame)

    # Camera intrinsics for 1920x1080 (approximate, refine with calibration)
    fx = fy = 1000.0  # Focal length (guess, adjust after calibration)
    cx, cy = 960.0, 540.0  # Principal point (center of 1920x1080)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # No distortion assumed

    # Tag size and map size in meters
    tag_size = 0.130175  # 5⅛ inches
    map_size = 1.502475  # 59⅛ inches

    # Define 3D tag corners in object coordinate system (square tag centered at origin)
    obj_points = np.array([
        [-tag_size/2, -tag_size/2, 0],  # Bottom-left
        [ tag_size/2, -tag_size/2, 0],  # Bottom-right
        [ tag_size/2,  tag_size/2, 0],  # Top-right
        [-tag_size/2,  tag_size/2, 0]   # Top-left
    ], dtype=np.float32)

    # Store tag poses
    tag_poses = {}
    for detection in detections:
        tag_id = detection.getId()
        corners = detection.getCorners((-1, -1, -1, -1, -1, -1, -1, -1))  # 8 values: x0, y0, x1, y1, ...
        img_points = np.array([
            [corners[0], corners[1]],  # Bottom-left
            [corners[2], corners[3]],  # Bottom-right
            [corners[4], corners[5]],  # Top-right
            [corners[6], corners[7]]   # Top-left
        ], dtype=np.float32)

        # Estimate pose with solvePnP
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if ret:
            # Convert rotation vector to matrix
            rmat, _ = cv2.Rodrigues(rvec)
            tag_poses[tag_id] = {'translation': tvec.flatten(), 'rotation': rmat}

        # Draw on the preprocessed frame (color version) for visualization
        center = detection.getCenter()
        print(f"Detected tag ID: {tag_id}, Center: ({center.x:.1f}, {center.y:.1f}) pixels")
        pt1 = (int(corners[0]), int(corners[1]))
        pt2 = (int(corners[4]), int(corners[5]))
        cv2.rectangle(processed_frame_display, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(processed_frame_display, str(tag_id), (int(center.x) + 10, int(center.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Check if all required tags are detected
    required_ids = {1, 2, 3, 4, 5}  # Top-left:1, Top-right:2, Bottom-left:3, Bottom-right:4, Car:5
    detected_ids = set(tag_poses.keys())
    missing_ids = required_ids - detected_ids
    if missing_ids:
        print(f"Warning: Missing tags {missing_ids}. Cannot calculate car position.")
    else:
        # Define coordinate system with tag 3 (bottom-left) as origin
        origin = tag_poses[3]['translation']
        t_car = tag_poses[5]['translation']  # Car

        # Transform to relative coordinates
        t_car_rel = t_car - origin

        # Define map axes (simplified assumption)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])

        # Project car position onto map axes
        x = np.dot(t_car_rel, x_axis)
        y = np.dot(t_car_rel, y_axis)

        # Clip to map bounds
        x = np.clip(x, 0, map_size)
        y = np.clip(y, 0, map_size)

        print(f"Car location on map: ({x:.3f}, {y:.3f}) meters")

        # Update the occupancy grid with the car's position
        occupancy_grid = create_occupancy_grid(map_size, CELL_SIZE_METERS)
        occupancy_grid = update_occupancy_grid(occupancy_grid, x, y, CELL_SIZE_METERS)

        # Optional: Visualize the occupancy grid (simple scaling for display)
        grid_display = cv2.resize(occupancy_grid * 255, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Occupancy Grid', grid_display)

    # Resize and display the preprocessed frame with detections
    scale_factor = 0.5
    resized_frame = cv2.resize(processed_frame_display, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    cv2.namedWindow('AprilTag Detection (Preprocessed)', cv2.WINDOW_NORMAL)
    cv2.imshow('AprilTag Detection (Preprocessed)', resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()