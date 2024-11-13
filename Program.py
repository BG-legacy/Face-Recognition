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

# Initialize the face recognizer and the Haar Cascade
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained model if it exists
try:
    recognizer.read('trainer.yml')
    with open("labels.pickle", 'rb') as f:
        labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}
except:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}

# Helper function to capture user images
from CaptureUserImages import capture_user_images

# Function to train the recognizer
def train_recognizer():
    label_id = {}
    face_train = []
    face_label = []
    current_id = 0
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    my_face_dir = os.path.join(BASE_DIR, 'image_data')

    image_files = []
    for root, dirs, files in os.walk(my_face_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    for path in image_files:
        label = os.path.basename(os.path.dirname(path)).lower()
        if label not in label_id:
            label_id[label] = current_id
            current_id += 1
        ID = label_id[label]

        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image, "uint8")
        faces = cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_train.append(image_array[y:y + h, x:x + w])
            face_label.append(ID)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_id, f)

    recognizer.train(face_train, np.array(face_label))
    recognizer.save("trainer.yml")
    print("Training completed and model saved.")

# Live video feed with face recognition
def recognize_face():
    cap = cv2.VideoCapture(0)
    
    # Initialize Arduino connection
    try:
        arduino = serial.Serial('/dev/cu.usbmodem1101', 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print("Arduino connection established")
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino: {e}")
        arduino = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces detected, send message to Arduino
        if len(faces) == 0 and arduino:
            try:
                arduino.write("User not identified\n".encode())
            except serial.SerialException as e:
                print(f"Failed to send data to Arduino: {e}")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 100:
                name = labels.get(label, "Unknown")
                message = "Welcome Home Boss" if name != "Unknown" else "User not identified"
                
                # Send message to Arduino
                if arduino:
                    try:
                        arduino.write(f"{message}\n".encode())
                    except serial.SerialException as e:
                        print(f"Failed to send data to Arduino: {e}")
            else:
                name = "Unknown"
                if arduino:
                    try:
                        arduino.write("User not identified\n".encode())
                    except serial.SerialException as e:
                        print(f"Failed to send data to Arduino: {e}")
            
            cv2.putText(frame, f"{name} - {round(100 - confidence)}%", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

# GUI Application using Tkinter
def start_gui():
    def capture_images():
        user_name = user_name_entry.get()
        if not user_name:
            messagebox.showerror("Error", "Please enter a name.")
            return
        try:
            num_images = int(num_images_entry.get())
            if num_images <= 0:
                messagebox.showerror("Error", "Please enter a positive number of images.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of images.")
            return
            
        capture_user_images(user_name, num_images)
        messagebox.showinfo("Info", f"Images captured for {user_name}.")

    def train_model():
        train_recognizer()
        messagebox.showinfo("Info", "Model training completed.")

    def recognize_from_camera():
        recognize_face()

    # Create the main window
    root = tk.Tk()
    root.title("Facial Recognition App")
    root.attributes("-topmost", True)
    

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

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    start_gui()
