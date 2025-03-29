import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetector, AprilTagPoseEstimator
from scipy.spatial.transform import Rotation

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    detector = AprilTagDetector()
    config = detector.Config()
    poseConfig = AprilTagPoseEstimator.Config(tagSize = 0.13, fx = 554, fy = 579, cx = 320, cy = 240)
    poseEstimator = AprilTagPoseEstimator(poseConfig)

    config.decodeSharpening = 0.75
    config.quadSigma = 0.4
    config.quadDecimate = 1

    detector.setConfig(config)
    detector.addFamily("tag36h11")

    previous_yaw = None  
    unwrapped_yaw = 0.0

    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale (needed for AprilTag detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Process', gray_frame)

        detections = detector.detect(gray_frame)

        for detection in detections:
            corner0 = detection.getCorner(0)  # Top-left
            corner2 = detection.getCorner(2)  # Bottom-right
            pt1 = (int(corner0.x), int(corner0.y))
            pt2 = (int(corner2.x), int(corner2.y))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)       

            #rot = poseEstimator.estimate(detection).rotation()
            #angle_degrees = np.sqrt(rot.x**2 + rot.y**2 + rot.z**2) * 180 / np.pi

            rot = poseEstimator.estimate(detection).rotation()
            rot_vec = np.array([rot.x, rot.y, rot.z])
            angle_degrees = np.linalg.norm(rot_vec) * 180 / np.pi

            # Convert to rotation matrix and then Euler angles
            rot_matrix, _ = cv2.Rodrigues(rot_vec)
            r = Rotation.from_matrix(rot_matrix)
            euler_degrees = r.as_euler('xyz', degrees=True)  
            current_yaw = euler_degrees[2]  

            if previous_yaw is not None:
                delta_yaw = current_yaw - previous_yaw
                if delta_yaw > 180:
                    delta_yaw -= 360
                elif delta_yaw < -180:
                    delta_yaw += 360
                unwrapped_yaw += delta_yaw
            else:
                unwrapped_yaw = current_yaw  # Initialize on first detection
            previous_yaw = current_yaw

            # Normalize to 0°–360°
            unwrapped_yaw = unwrapped_yaw % 360
            if unwrapped_yaw < 0:
                unwrapped_yaw += 360

            # text = f"ID: {detection.getId()}, Rot: {unwrapped_yaw:.1f}°"
            text_position = (pt1[0], pt1[1] - 10)  
            cv2.putText(
                frame,
                text,
                text_position,     
                cv2.FONT_HERSHEY_SIMPLEX,  
                0.5,               
                (0, 255, 0),       
                1,                 
                cv2.LINE_AA       
            )                

        # Display the frame with detected tags
        # cv2.imshow('AprilTag Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()