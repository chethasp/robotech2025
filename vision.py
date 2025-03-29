import cv2
from robotpy_apriltag import AprilTagDetector

def main():
    # Open the camera (0 is usually the default camera)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Create an AprilTag detector
    detector = AprilTagDetector()
    config = detector.Config()

    config.decodeSharpening = 0.75
    config.quadSigma = 0.4
    config.quadDecimate = 1

    detector.setConfig(config)
    detector.addFamily("tag36h11")

    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale (needed for AprilTag detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Process', gray_frame)

        # Detect AprilTags in the grayscale frame
        detections = detector.detect(gray_frame)

        # Draw detections on the frame
        for detection in detections:
            print(f"Detected tag ID: {detection.getId()}")

            
            print(f"Tag center: {detection.getCenter()}")
            # Optionally, draw a rectangle around the tag (you can modify the detection area here)
            corner0 = detection.getCorner(0)  # Top-left
            corner2 = detection.getCorner(2)  # Bottom-right
            pt1 = (int(corner0.x), int(corner0.y))
            pt2 = (int(corner2.x), int(corner2.y))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)                          

        # Display the frame with detected tags
        cv2.imshow('AprilTag Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()