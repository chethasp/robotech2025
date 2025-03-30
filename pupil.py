from pupil_apriltags import Detector
import cv2
import numpy as np

# Initialize the detector with tag36h11 family
detector = Detector(families="tag36h11")

# Load the JPG image
image_path = "test.jpg"  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image from '{image_path}' liquid.")
    exit()

# Convert to grayscale before detection (required by pupil-apriltags)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Camera parameters (replace with your calibrated values)
camera_params = [fx, fy, cx, cy] = [1000.0, 1000.0, 960.0, 540.0]  # Updated for 1920x1080
tag_size = 0.130175  # Tag width in meters (5 1/8 inches)
map_size = 1.502475  # Map side length in meters (59 1/8 inches)

# Detect tags with pose estimation using grayscale image
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

# Print detected tag IDs
detected_ids = [detection.tag_id for detection in detections]
print(f"Detected tag IDs: {detected_ids}")

# Store poses by tag ID
tag_poses = {}
for detection in detections:
    tag_poses[detection.tag_id] = {
        'translation': detection.pose_t.flatten(),  # [x, y, z] in meters
        'rotation': detection.pose_R               # 3x3 rotation matrix
    }

# Check if all required tags are detected
required_ids = {1, 2, 3, 4, 5}  # Top-left:1, Top-right:2, Bottom-left:3, Bottom-right:4, Car:5
missing_ids = required_ids - set(detected_ids)
if missing_ids:
    print(f"Warning: Missing tags {missing_ids}. Proceeding with detected tags.")
else:
    # Define the world coordinate system with tag 3 (bottom-left) as origin
    origin = tag_poses[3]['translation']  # Bottom-left (ID 3)
    t1 = tag_poses[1]['translation']  # Top-left
    t2 = tag_poses[2]['translation']  # Top-right
    t3 = tag_poses[3]['translation'] #Bottom-left
    t4 = tag_poses[4]['translation']  # Bottom-right
    t_car = tag_poses[5]['translation']  # Car

    # Transform all poses to be relative to tag 3 (bottom-left)
    t1_rel = t1 - t3
    t2_rel = t2 - t3
    t4_rel = t4 - t3
    t_car_rel = t_car - t3

    # Define map axes
    x_axis = t4_rel / np.linalg.norm(t4_rel)  # X-axis: tag 3 to tag 4
    y_axis_raw = t1_rel - np.dot(t1_rel, x_axis) * x_axis  # Y-axis: tag 3 to tag 1, orthogonalized
    y_axis = y_axis_raw / np.linalg.norm(y_axis_raw)

    # Project car position onto map axes
    x = np.dot(t_car_rel, x_axis)
    y = np.dot(t_car_rel, y_axis)

    # Ensure coordinates are within map bounds
    x = np.clip(x, 0, map_size)
    y = np.clip(y, 0, map_size)

    print(f"Car location on map: ({x:.3f}, {y:.3f}) meters")

# Draw detections on the original color image
for detection in detections:
    corners = detection.corners.astype(int)
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i+1) % 4]), (0, 255, 0), 2)
    center = tuple(detection.center.astype(int))
    cv2.circle(img, center, 5, (0, 0, 255), -1)
    cv2.putText(img, str(detection.tag_id), (center[0] + 10, center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Convert to grayscale after detection and drawing for display
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image to make the window smaller (50% of original size)
scale_factor = 0.5
resized_gray_img = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Create a resizable window and display the resized grayscale image
cv2.namedWindow('Map with Detections (Grayscale)', cv2.WINDOW_NORMAL)
cv2.imshow('Map with Detections (Grayscale)', resized_gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()