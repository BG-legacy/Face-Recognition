import os
import cv2
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

def getData():
    label_id = {}  # Dictionary to store label-to-id mappings
    face_train = []  # List to store face image arrays
    face_label = []  # List to store corresponding face labels
    current_id = 0
    # Load the Haar Cascade classifier for face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    if cascade.empty():
        raise IOError("Unable to load the face cascade classifier xml file")

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
        print(f"Contents of image_data directory: {os.listdir(my_face_dir)}")
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

        # Enhanced preprocessing pipeline
        def preprocess_image(image_array):
            # Enhance contrast
            image_array = cv2.equalizeHist(image_array)
            
            # Apply bilateral filter to reduce noise while preserving edges
            image_array = cv2.bilateralFilter(image_array, 9, 75, 75)
            
            # Normalize the image
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image_array = clahe.apply(image_array)
            
            return image_array

        # Improved face detection parameters
        faces = cascade.detectMultiScale(
            image_array,
            scaleFactor=1.1,
            minNeighbors=8,       # Increased to reduce false positives
            minSize=(100, 100),   # Increased minimum face size
            maxSize=(500, 500),   # Maximum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Add face validation function
        def is_valid_face(face_roi):
            # Check aspect ratio
            height, width = face_roi.shape
            aspect_ratio = width / height
            if not (0.5 <= aspect_ratio <= 1.5):
                return False
            
            # Check minimum contrast
            min_val, max_val, _, _ = cv2.minMaxLoc(face_roi)
            if (max_val - min_val) < 30:
                return False
            
            # Check face symmetry
            left_half = face_roi[:, :width//2]
            right_half = cv2.flip(face_roi[:, width//2:], 1)
            symmetry_score = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            if symmetry_score < 0.5:  # Adjust threshold as needed
                return False
            
            return True

        # Process only valid faces
        valid_faces = []
        for (x, y, w, h) in faces:
            face_roi = image_array[y:y+h, x:x+w]
            if is_valid_face(face_roi):
                valid_faces.append((x, y, w, h))
        
        for (x, y, w, h) in valid_faces:
            roi = image_array[y:y + h, x:x + w]
            processed_face = preprocess_image(roi)
            processed_face = cv2.resize(processed_face, (200, 200))
            face_train.append(processed_face)
            face_label.append(ID)

    # Save the label-to-id mappings
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_id, f)

    return face_train, face_label

def augment_face(face_image):
    augmented_faces = []
    
    # Original image
    augmented_faces.append(face_image)
    
    # Slightly rotated versions
    for angle in [-5, 5]:
        matrix = cv2.getRotationMatrix2D((face_image.shape[1]/2, face_image.shape[0]/2), angle, 1.0)
        rotated = cv2.warpAffine(face_image, matrix, (face_image.shape[1], face_image.shape[0]))
        augmented_faces.append(rotated)
    
    # Slightly scaled versions
    for scale in [0.95, 1.05]:
        scaled = cv2.resize(face_image, None, fx=scale, fy=scale)
        scaled = cv2.resize(scaled, (face_image.shape[1], face_image.shape[0]))
        augmented_faces.append(scaled)
    
    return augmented_faces

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Increase radius for more detail
        neighbors=12,       # More sampling points
        grid_x=10,         # More cells horizontally
        grid_y=10,         # More cells vertically
        threshold=80.0     # Lower threshold for stricter matching
    )

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
        print("Training completed and model saved.")

if __name__ == "__main__":
    train_recognizer()

# Resources:
# 1. OpenCV Face Recognition: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
# 2. LBPH Face Recognizer: https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html
# 3. Face Recognition with Python: https://realpython.com/face-recognition-with-python/
# 4. Haar Cascade Object Detection: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# 5. Working with Images in Python: https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
# 6. NumPy for Image Processing: https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-use-numpy-for-image-processing
# 7. Python os module: https://docs.python.org/3/library/os.html
# 8. Python pickle module: https://docs.python.org/3/library/pickle.html
