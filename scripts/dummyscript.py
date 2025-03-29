import cv2
from vision import get_robot_heading, get_current_frame
from realpathplanning import create_path
from occupanyGrid import gen_occupany

# print(type(get_current_frame()))
# image = cv2.imread("images/grid.png")
# print(type(image))

camera = cv2.VideoCapture(1)  # Open camera once

if not camera.isOpened():
    print("Error: Couldn’t open the camera.")
    exit()
    
while True:
    img = get_current_frame(camera)
    if img is None:
        break  # Exit if frame grab fails
    
    path = create_path(img)  # Process the frame (your occupancy grid + path)
    occ = gen_occupany(img)  # Process the frame (your occupancy grid + path)
    # image = cv2.imread("images/grid1.jpg")
    
    # path = create_path(image)
    # Display the result
    cv2.imshow("path1", path)
    # print(occ)
    cv2.imshow("occ", occ)
    cv2.imshow("img", img)

    # Wait for 1ms, quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()