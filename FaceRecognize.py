import cv2
import pickle
import logging
import os
import serial
import time

# Configure logging to track events and errors
logging.basicConfig(level=logging.INFO)

# Constants for file paths and face recognition parameters
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"
CONFIDENCE_THRESHOLD_MIN = 20
CONFIDENCE_THRESHOLD_MAX = 85
ARDUINO_PORT = '/dev/cu.usbmodem1101'  # Updated to correct port name
BAUD_RATE = 9600

# Initialize video capture from default camera
video = cv2.VideoCapture(0)

try:
    # Load the Haar Cascade Classifier for face detection
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise IOError("Unable to load the face cascade classifier xml file")

    # Create and load the LBPH Face Recognizer
    recognise = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognise.read(TRAINER_PATH)
    except cv2.error as e:
        print(f"Error loading recognizer model: {e}")

    # Load labels from pickle file
    with open(LABELS_PATH, 'rb') as f:
        og_label = pickle.load(f)
        labels = {v: k for k, v in og_label.items()}
    logging.info(f"Loaded labels: {labels}")

    # Initialize Arduino serial connection
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    logging.info("Arduino connection established")

    # Fix: Move the while loop inside the try block (fix indentation)
    while True:
        # Capture frame-by-frame
        check, frame = video.read()
        if not check:
            logging.warning("Failed to capture frame")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame with stricter parameters
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,    # Decreased from 1.2 for more precise detection
            minNeighbors=8,     # Increased from 5 for stricter detection
            minSize=(30, 30)    # Add minimum face size requirement
        )

        # Add this: If no faces detected, send message to Arduino
        if len(faces) == 0 and arduino:
            try:
                arduino.write("User not identified\n".encode())
                logging.info("No face detected")
            except serial.SerialException as e:
                logging.error(f"Failed to send data to Arduino: {e}")

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
                if arduino:
                    try:
                        arduino.write(f"{message}\n".encode())
                        logging.info(f"Sent to Arduino: {message}")
                    except serial.SerialException as e:
                        logging.error(f"Failed to send data to Arduino: {e}")

                # Draw the name on the frame
                cv2.putText(frame, name, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (18,5,255), 2, cv2.LINE_AA)
            else:
                # If confidence is low, send "User not identified" to Arduino
                pass  # Commented Arduino code here

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)

        # Display the resulting frame
        cv2.imshow("Video", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except serial.SerialException as e:
    logging.error(f"Failed to connect to Arduino: {e}")
    arduino = None
except IOError as e:
    logging.error(f"File error: {e}")
except Exception as e:
    logging.error(f"An error occurred: {e}")
finally:
    # Clean up resources
    video.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

# Resources for further learning:
# 1. OpenCV Face Recognition: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
# 2. PyImageSearch Face Recognition: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# 3. Real Python Face Recognition: https://realpython.com/face-recognition-with-python/
# 4. Arduino Serial Communication: https://www.arduino.cc/en/Tutorial/BuiltInExamples/SerialEvent
# 5. Python Serial Library: https://pythonhosted.org/pyserial/
# 6. Haar Cascade Classifiers: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# 7. LBPH Face Recognizer: https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html
