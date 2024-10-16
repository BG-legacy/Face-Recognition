import os
import cv2
import numpy as np
from PIL import Image
import pickle

def getData():
    label_id = {}  # Dictionary to store label-to-id mappings
    face_train = []  # List to store face image arrays
    face_label = []  # List to store corresponding face labels
    current_id = 0
    # Load the Haar Cascade classifier for face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Get the directory of the current script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Set the path to the directory containing the face images
    my_face_dir = os.path.join(BASE_DIR, 'image_data')

    # Debugging: Print the full path to ensure it's correct
    print(f"Image directory full path: {os.path.abspath(my_face_dir)}")

    # Check if the directory exists
    if not os.path.exists(my_face_dir):
        print(f"Directory does not exist: {my_face_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of current directory: {os.listdir()}")
        return [], []

    # Recursively find all image files in the directory and its subdirectories
    image_files = []
    for root, dirs, files in os.walk(my_face_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No image files found in {my_face_dir} or its subdirectories")
        print(f"Contents of image_data directory: {os.listdir(my_face_dir)}")
        return [], []

    print(f"Found {len(image_files)} image files")

    # Process each image file
    for path in image_files:
        # Extract the label (person's name) from the parent folder name
        label = os.path.basename(os.path.dirname(path)).lower()

        # Assign a unique ID to each label
        if label not in label_id:
            label_id[label] = current_id
            current_id += 1
        ID = label_id[label]

        # Open the image, convert to grayscale, and create a numpy array
        pil_image = Image.open(path).convert("L")  # convert image to grayscale
        image_array = np.array(pil_image, "uint8")  # convert image to numpy array

        # Detect faces in the image
        faces = cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"No faces detected in {path}")
        else:
            for (x, y, w, h) in faces:
                # Extract the face region and add to training data
                face_train.append(image_array[y:y + h, x:x + w])
                face_label.append(ID)

    # Save the label-to-id mappings
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_id, f)

    return face_train, face_label

# Create a LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Get the training data
faces, ids = getData()

# Debugging: Print the number of faces and ids collected
print(f"Number of faces detected: {len(faces)}")
print(f"Number of IDs detected: {len(ids)}")

if len(faces) == 0 or len(ids) == 0:
    print("No training data found. Ensure that images contain faces and are in the correct directory.")
else:
    # Train the recognizer
    recognizer.train(faces, np.array(ids))
    # Save the trained model
    recognizer.save("trainer.yml")

# Resources for further learning:
# 1. OpenCV Face Recognition: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
# 2. LBPH Face Recognizer: https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html
# 3. Face Recognition with Python: https://realpython.com/face-recognition-with-python/
# 4. Haar Cascade Object Detection: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# 5. Working with Images in Python: https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
# 6. NumPy for Image Processing: https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-use-numpy-for-image-processing
# 7. Python os module: https://docs.python.org/3/library/os.html
# 8. Python pickle module: https://docs.python.org/3/library/pickle.html
