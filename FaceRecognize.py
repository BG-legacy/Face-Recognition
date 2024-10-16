import cv2
import pickle
import logging
import os
import serial.tools.list_ports
from serial import Serial
import time

# Configure logging to track events and errors
logging.basicConfig(level=logging.INFO)

# Constants for file paths and face recognition parameters
CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.yml")
LABELS_PATH = "labels.pickle"
CONFIDENCE_THRESHOLD_MIN = 20
CONFIDENCE_THRESHOLD_MAX = 115

# Initialize serial communication with Arduino
# Note: Replace '/dev/tty.usbmodem2101' with your actual port
arduino = Serial('/dev/tty.usbmodem2101', 9600)

# Give time for Arduino to reset (optional)
time.sleep(2)

print("Connected to Arduino!")

try:
    # Load the Haar Cascade Classifier for face detection
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise IOError("Unable to load the face cascade classifier xml file")

    # Create and load the LBPH Face Recognizer
    recognise = cv2.face.LBPHFaceRecognizer_create()
    recognise.read(TRAINER_PATH)

    # Load labels from pickle file
    with open(LABELS_PATH, 'rb') as f:
        og_label = pickle.load(f)
        labels = {v: k for k, v in og_label.items()}
    logging.info(f"Loaded labels: {labels}")

    # Initialize video capture from default camera
    video = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        check, frame = video.read()
        if not check:
            logging.warning("Failed to capture frame")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for x, y, w, h in faces:
            # Extract the face ROI (Region of Interest)
            face_save = gray[y:y+h, x:x+w]
            # Predict the face using the trained recognizer
            ID, conf = recognise.predict(face_save)

            if CONFIDENCE_THRESHOLD_MIN <= conf <= CONFIDENCE_THRESHOLD_MAX:
                # If confidence is within acceptable range, get the name
                name = labels.get(ID, "Unknown")
                message = "Welcome Home Boss" if name != "Unknown" else "User not identified"
                logging.info(f"Recognized: {name} (ID: {ID}, Confidence: {conf})")

                # Send message to Arduino
                try:
                    arduino.write((message + '\n').encode())
                    arduino.flush()  # Ensure the message is sent immediately
                except serial.SerialException as e:
                    logging.error(f"Failed to send message to Arduino: {e}")

                # Draw the name on the frame
                cv2.putText(frame, name, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (18,5,255), 2, cv2.LINE_AA)
            else:
                # If confidence is low, send "User not identified" to Arduino
                try:
                    arduino.write("User not identified\n".encode())
                    arduino.flush()
                except serial.SerialException as e:
                    logging.error(f"Failed to send message to Arduino: {e}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)

        # Display the resulting frame
        cv2.imshow("Video", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except IOError as e:
    logging.error(f"File error: {e}")
except Exception as e:
    logging.error(f"An error occurred: {e}")
finally:
    # Clean up resources
    video.release()
    cv2.destroyAllWindows()
    arduino.close()

# Resources for further learning:
# 1. OpenCV Face Recognition: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
# 2. PyImageSearch Face Recognition: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# 3. Real Python Face Recognition: https://realpython.com/face-recognition-with-python/
# 4. Arduino Serial Communication: https://www.arduino.cc/en/Tutorial/BuiltInExamples/SerialEvent
# 5. Python Serial Library: https://pythonhosted.org/pyserial/
# 6. Haar Cascade Classifiers: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# 7. LBPH Face Recognizer: https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html
