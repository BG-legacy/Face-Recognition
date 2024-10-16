import cv2  # OpenCV library for computer vision tasks
import time  # For adding delays

# Open the default webcam (index 0)
video = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Add a small delay to ensure the webcam is initialized
# This can help prevent issues with some webcams
time.sleep(2)

while True:
    # Capture frame-by-frame
    # 'check' is a boolean indicating if the frame was successfully read
    # 'frame' is the captured image as a numpy array
    check, frame = video.read()
    
    # Check if the frame is read correctly
    if not check or frame is None:
        print("Error: Could not read frame.")
        break
    
    # Display the resulting frame
    cv2.imshow("video", frame)
    
    # Wait for 1 millisecond and capture any key press
    key = cv2.waitKey(1)
    
    # If 'q' is pressed, break the loop
    if key == ord('q'):
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Resources for further learning:
# 1. OpenCV Documentation: https://docs.opencv.org/
# 2. OpenCV-Python Tutorials: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# 3. Real Python - Face Detection in Python Using a Webcam: https://realpython.com/face-detection-in-python-using-a-webcam/
# 4. PyImageSearch - OpenCV Tutorial: A Guide to Learn OpenCV: https://pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/
# 5. Towards Data Science - Introduction to Video Capture with OpenCV and Python: https://towardsdatascience.com/introduction-to-video-capture-with-opencv-and-python-48c2a3ea9b0c
