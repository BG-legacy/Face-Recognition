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

# Load the Haar Cascade Classifier for face detection
# This XML file contains the pre-trained model for detecting frontal faces
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    # 'check' is a boolean indicating if the frame was successfully read
    # 'frame' is the captured image as a numpy array
    check, frame = video.read()
    
    # Check if the frame is read correctly
    if not check or frame is None:
        print("Error: Could not read frame.")
        break
    
    # Convert the frame to grayscale
    # Face detection works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    # detectMultiScale returns a list of (x, y, w, h) tuples for each detected face
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    
    # Draw rectangles around the detected faces
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
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
# 1. OpenCV Face Detection Tutorial: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# 2. Haar Cascade Classifiers: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# 3. Real Python - Face Detection in Python Using a Webcam: https://realpython.com/face-detection-in-python-using-a-webcam/
# 4. PyImageSearch - Face detection with OpenCV and deep learning: https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
# 5. Towards Data Science - Face Detection with Haar Cascade: https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08
# 6. OpenCV-Python Tutorials: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
