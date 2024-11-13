# Import section
import cv2                  # OpenCV library for computer vision tasks and image processing
import os                   # Operating system module for file/directory handling
import time                 # Time module for adding delays between operations

def capture_user_images(user_name, num_images=5):
    # Create path by joining 'image_data' directory with user_name using os.path.join for cross-platform compatibility
    user_dir = os.path.join('image_data', user_name)
    
    # Check if user's directory already exists to prevent overwriting
    if os.path.exists(user_dir):
        # Prompt user for permission to add more photos if directory exists
        response = input(f"Folder for {user_name} already exists. Do you want to add more photos? (yes/no): ").strip().lower()
        # If user doesn't confirm with 'yes', exit the function
        if response != 'yes':
            print("Exiting without capturing images.")
            return
    else:
        # Create new directory if it doesn't exist, exist_ok=True prevents errors if directory is created between check and creation
        os.makedirs(user_dir, exist_ok=True)

    # Initialize video capture object using default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    # Verify camera initialization was successful
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load pre-trained face detection model from OpenCV's Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Display instructions for the user
    print(f"Capturing {num_images} images for {user_name}. Position your face in the box and press SPACE to capture.")

    # Initialize list to store existing image numbers
    existing_images = []
    # Iterate through all files in user's directory
    for f in os.listdir(user_dir):
        # Check if file is a jpg image
        if f.endswith('.jpg'):
            try:
                # First attempt: try parsing simple number format
                num = int(f.split('.')[0])
            except ValueError:
                try:
                    # Second attempt: try parsing user_name_number format
                    num = int(f.split('_')[1].split('.')[0])
                except (ValueError, IndexError):
                    # Skip file if neither format matches
                    continue
            # Add successfully parsed number to list
            existing_images.append(num)
    
    # Determine starting index for new images (continue from highest existing number or start at 0)
    start_index = max(existing_images) + 1 if existing_images else 0

    # Main capture loop
    while True:
        # Initialize counter for current capture session
        count = 0
        # Loop until requested number of images are captured
        while count < num_images:
            # Capture single frame from video feed
            ret, frame = cap.read()
            # Check if frame was successfully captured
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in grayscale image using cascade classifier
            # scaleFactor: how much image size is reduced at each scale
            # minNeighbors: how many neighbors each candidate rectangle should have
            # minSize: minimum possible face size to detect
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Flag to track if any face is detected in current frame
            face_detected = False
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw green rectangle around detected face (BGR color format)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_detected = True

            # Display frame with face detection rectangles
            cv2.imshow("Capture", frame)

            # Check for keyboard input (1ms wait)
            key = cv2.waitKey(1)
            # If space pressed and face detected, save image
            if key == ord(' ') and face_detected:
                # Generate unique filename using user_name and current count
                image_path = os.path.join(user_dir, f"{user_name}_{start_index + count}.jpg")
                # Save current frame as jpg image
                cv2.imwrite(image_path, frame)
                count += 1
                # Delay to prevent accidental multiple captures
                time.sleep(1)
            # If 'q' pressed, exit capture loop
            elif key == ord('q'):
                break

        # Ask user if they want to capture more images after completing current set
        more_images = input("Do you want to capture more images? (yes/no): ").strip().lower()
        if more_images != 'yes':
            break
        else:
            # Get new number of images to capture
            num_images = int(input("Enter the number of additional images to capture: "))

    # Release camera resources
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    # Print final capture count
    print(f"Captured {count} images for {user_name}")

# Script entry point
if __name__ == "__main__":
    # Get user input for name and number of images
    user_name = input("Enter your name: ")
    num_images = int(input("Enter the number of images to capture: "))
    # Call main function with user inputs
    capture_user_images(user_name, num_images)
