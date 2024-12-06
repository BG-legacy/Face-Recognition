import cv2
import os
import numpy as np
from PIL import Image
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import time
import serial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the face recognizer and the Haar Cascade
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained model if it exists
try:
    # Get absolute path to current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    trainer_path = os.path.join(BASE_DIR, 'trainer.yml')
    labels_path = os.path.join(BASE_DIR, 'labels.pickle')
    
    logger.info(f"Looking for trainer file at: {trainer_path}")
    logger.info(f"Looking for labels file at: {labels_path}")
    
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"Trainer file not found at {trainer_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    # Load the recognizer
    recognizer.read(trainer_path)
    logger.info("Successfully loaded trainer.yml")
    
    # Load the labels
    with open(labels_path, 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
    logger.info(f"Successfully loaded labels: {labels}")
    
except FileNotFoundError as e:
    logger.error(f"File not found error: {e}")
    logger.info("Please ensure you have run the training process first")
    # Initialize empty recognizer and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}
except Exception as e:
    logger.error(f"Error loading training data: {e}")
    # Initialize empty recognizer and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}

# Helper function to capture user images
from CaptureUserImages import capture_user_images

# Function to train the recognizer
def train_recognizer(update_status):
    try:
        logger.info("Starting training process...")
        
        # Get absolute path to image_data directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_data_dir = os.path.join(BASE_DIR, 'image_data')
        
        logger.info(f"Looking for images in: {image_data_dir}")
        
        if not os.path.exists(image_data_dir):
            logger.error(f"image_data directory not found at {image_data_dir}")
            raise Exception("image_data directory not found")
            
        # List all subdirectories (person names)
        person_dirs = [d for d in os.listdir(image_data_dir) 
                      if os.path.isdir(os.path.join(image_data_dir, d))]
        
        logger.info(f"Found person directories: {person_dirs}")
        
        if not person_dirs:
            logger.error("No person directories found in image_data")
            raise Exception("No person directories found")
            
        label_id = {}
        face_train = []
        face_label = []
        current_id = 0
        
        # Process each person's directory
        for person_dir in person_dirs:
            person_path = os.path.join(image_data_dir, person_dir)
            logger.info(f"Processing images for {person_dir}")
            
            # Add person to label_id dictionary
            if person_dir not in label_id:
                label_id[person_dir] = current_id
                current_id += 1
            
            # Process each image in the person's directory
            for image_file in os.listdir(person_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    logger.info(f"Processing image: {image_path}")
                    
                    # Read and convert image to grayscale
                    pil_image = Image.open(image_path).convert("L")
                    image_array = np.array(pil_image, "uint8")
                    
                    # Detect faces in the image
                    faces = cascade.detectMultiScale(
                        image_array,
                        scaleFactor=1.1,
                        minNeighbors=8,
                        minSize=(100, 100),
                        maxSize=(500, 500)
                    )
                    
                    # Process detected faces
                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        roi = cv2.resize(roi, (200, 200))  # Resize to consistent size
                        
                        # Enhance the ROI
                        roi = cv2.equalizeHist(roi)  # Enhance contrast
                        roi = cv2.GaussianBlur(roi, (5, 5), 0)  # Reduce noise
                        
                        # Add to training data
                        face_train.append(roi)
                        face_label.append(label_id[person_dir])
                        
                        logger.info(f"Added face for {person_dir} with label {label_id[person_dir]}")
        
        logger.info(f"Total faces collected: {len(face_train)}")
        logger.info(f"Labels created: {label_id}")
        
        if len(face_train) == 0:
            raise Exception("No faces were detected in the training images")
            
        # Convert lists to numpy arrays
        face_train = np.array(face_train)
        face_label = np.array(face_label)
        
        # Train the recognizer
        recognizer.train(face_train, face_label)
        
        # Save the trained model
        trainer_path = os.path.join(BASE_DIR, 'trainer.yml')
        labels_path = os.path.join(BASE_DIR, 'labels.pickle')
        
        recognizer.save(trainer_path)
        with open(labels_path, "wb") as f:
            pickle.dump(label_id, f)
            
        logger.info("Training completed successfully")
        logger.info(f"Model saved to: {trainer_path}")
        logger.info(f"Labels saved to: {labels_path}")
        
        # Verify the files were created
        if not os.path.exists(trainer_path):
            raise Exception("Failed to save trainer.yml")
        if not os.path.exists(labels_path):
            raise Exception("Failed to save labels.pickle")
            
        # Reload the recognizer and labels
        global labels
        recognizer.read(trainer_path)
        with open(labels_path, 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}
        
        logger.info("Model and labels reloaded successfully")
        logger.info(f"Final labels: {labels}")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

# Add this after training to verify the model was saved
def verify_training():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    trainer_path = os.path.join(BASE_DIR, 'trainer.yml')
    labels_path = os.path.join(BASE_DIR, 'labels.pickle')
    
    if not os.path.exists(trainer_path):
        logger.error("trainer.yml not found after training!")
        return False
    
    if not os.path.exists(labels_path):
        logger.error("labels.pickle not found after training!")
        return False
    
    try:
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
            logger.info(f"Labels verified: {labels}")
        logger.info("Training files verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying training files: {e}")
        return False

# Add this function before the recognize_face function

def preprocess_image(image):
    # Resize to a standard size
    image = cv2.resize(image, (100, 100))
    # Apply histogram equalization to improve contrast
    image = cv2.equalizeHist(image)
    return image

# Live video feed with face recognition
def recognize_face(update_status):
    cap = cv2.VideoCapture(0)
    
    if not labels:
        update_status("Error: No training data loaded. Please train the model first.", True)
        return

    update_status("Starting face recognition...")
    update_status(f"Loaded models for: {', '.join(labels.values())}")

    # Initialize Arduino connection
    try:
        arduino = serial.Serial('/dev/cu.usbmodem2101', 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        update_status("Arduino connected successfully")
    except serial.SerialException as e:
        update_status(f"Arduino connection failed: {str(e)}", True)
        arduino = None

    last_sent_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            update_status("Error: Cannot read from camera", True)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(100, 100),
            maxSize=(500, 500)
        )

        # Process detected faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            try:
                # Preprocess ROI
                roi_gray = cv2.equalizeHist(roi_gray)
                roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                
                # Get prediction
                label, confidence = recognizer.predict(roi_gray)
                confidence_score = max(0, min(100, (100 - confidence)))

                # Handle prediction results
                if label < 0 or confidence > 100:
                    message = "User not identified"
                    display_confidence = 0
                else:
                    if confidence_score < 60:
                        message = "User not identified"
                        display_confidence = 0
                    else:
                        name = labels.get(label, "Unknown")
                        message = "Welcome Home"
                        display_confidence = confidence_score
                
                # Send message to Arduino if it's different from last message
                if arduino and message != last_sent_message:
                    arduino.write(f"{message}\n".encode())
                    last_sent_message = message
                    update_status(f"Sent to Arduino: {message}")
                
                # Display results on video feed
                color = (0, 255, 0) if display_confidence > 60 else (0, 0, 255)
                text = f"{labels.get(label, 'Unknown')} - {int(display_confidence)}%" if display_confidence > 60 else "User not identified"
                cv2.putText(frame, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
            except Exception as e:
                update_status(f"Recognition error: {str(e)}", True)
                continue

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    update_status("Face recognition stopped")

# Add this function to verify the model is loaded correctly
def verify_model_loaded():
    try:
        # Get absolute path to current directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        trainer_path = os.path.join(BASE_DIR, 'trainer.yml')
        labels_path = os.path.join(BASE_DIR, 'labels.pickle')
        
        if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
            logger.error("Training files not found")
            return False
            
        # Try to load a test image and predict
        test_image = np.zeros((200, 200), dtype=np.uint8)  # Create dummy image
        try:
            label, conf = recognizer.predict(test_image)
            logger.info("Model can make predictions")
            return True
        except Exception as e:
            logger.error(f"Model prediction test failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying model: {e}")
        return False

def debug_training():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_data_dir = os.path.join(BASE_DIR, 'image_data')
    
    logger.info(f"Checking image_data directory: {image_data_dir}")
    if not os.path.exists(image_data_dir):
        logger.error("image_data directory not found!")
        return
        
    # Check directories and images
    for person_dir in os.listdir(image_data_dir):
        person_path = os.path.join(image_data_dir, person_dir)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"Found {len(images)} images for {person_dir}")
            
            # Check first image
            if images:
                test_image_path = os.path.join(person_path, images[0])
                img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
                faces = cascade.detectMultiScale(
                    img,
                    scaleFactor=1.1,
                    minNeighbors=8,
                    minSize=(100, 100),
                    maxSize=(500, 500)
                )
                logger.info(f"Detected {len(faces)} faces in test image for {person_dir}")

# GUI Application using Tkinter
def start_gui():
    def update_status(message, is_error=False):
        status_text.config(state='normal')
        status_text.insert(tk.END, f"\n{message}")
        if is_error:
            status_text.tag_add("error", "end-1c linestart", "end-1c")
        status_text.see(tk.END)
        status_text.config(state='disabled')

    def train_model():
        try:
            train_recognizer(update_status)
            if verify_training():
                update_status("Model training completed and verified successfully")
            else:
                update_status("Training completed but verification failed", True)
        except Exception as e:
            update_status(f"Training failed: {str(e)}", True)
    
    def recognize_from_camera():
        if not verify_model_loaded():
            update_status("Error: Face recognition model not properly loaded. Please train the model first.", True)
            return
        # Pass update_status to recognize_face
        recognize_face(update_status)

    def capture_images():
        user_name = user_name_entry.get()
        if not user_name:
            update_status("Error: Please enter a name.", True)
            return
        try:
            num_images = int(num_images_entry.get())
            if num_images <= 0:
                update_status("Error: Please enter a positive number of images.", True)
                return
        except ValueError:
            update_status("Error: Please enter a valid number of images.", True)
            return
            
        update_status(f"Starting capture for {user_name}...")
        try:
            capture_user_images(user_name, num_images)
            update_status(f"Successfully captured {num_images} images for {user_name}")
        except Exception as e:
            update_status(f"Error capturing images: {str(e)}", True)

    def debug_training_data():
        debug_training()
        messagebox.showinfo("Debug", "Check console for training data debug information")
    
    # Create the main window
    root = tk.Tk()
    root.title("Facial Recognition App")
    root.attributes("-topmost", True)
    
    # Add this line to create the status text widget
    global status_text
    status_text = tk.Text(root, height=10, width=50)
    status_text.pack(pady=10)
    status_text.config(state='disabled')
    
    # User input for name and buttons for different actions
    tk.Label(root, text="Enter your name:").pack()
    user_name_entry = tk.Entry(root)
    user_name_entry.pack()

    tk.Label(root, text="Number of images to capture:").pack()
    num_images_entry = tk.Entry(root)
    num_images_entry.insert(0, "5")  # Default value
    num_images_entry.pack()

    capture_button = tk.Button(root, text="Capture Images", command=capture_images)
    capture_button.pack(pady=10)

    train_button = tk.Button(root, text="Train Model", command=train_model)
    train_button.pack(pady=10)

    recognize_button = tk.Button(root, text="Recognize Face", command=recognize_from_camera)
    recognize_button.pack(pady=10)

    # Add debug button
    debug_button = tk.Button(root, text="Debug Training Data", command=debug_training_data)
    debug_button.pack(pady=10)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    start_gui()
