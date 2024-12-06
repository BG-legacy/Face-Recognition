# Face Recognition Security System

A smart security system that uses facial recognition to control access and provide visual feedback through an Arduino-controlled LED and OLED display setup.

## Features

- Real-time face detection and recognition
- Arduino-based feedback system with LED indicators
- OLED display for status messages
- User image capture and training system
- Robust face detection with confidence scoring

## Hardware Requirements

- Arduino board (compatible with Arduino IDE)
- OLED Display (SSD1306)
- Red LED
- Green LED
- USB Webcam
- Connecting wires
- Breadboard (optional, for prototyping)

## Software Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- PySerial
- Pillow (PIL)
- Arduino IDE

## Installation

1. Clone this repository:
   bash
git clone [repository-url]

2. Install required Python packages:
   bash
pip install opencv-python numpy pyserial pillow


3. Install the Arduino IDE and required libraries:
   - Adafruit_GFX
   - Adafruit_SSD1306
   - Wire

## Project Structure

- `AccessToWebCam.py` - Basic webcam access test
- `CaptureUserImages.py` - Tool for capturing user images for training
- `FaceIdentification.py` - Basic face detection implementation
- `FaceRecognize.py` - Main face recognition system
- `FaceTrainer.py` - Training system for face recognition
- `sketch_oct15a.ino` - Arduino code for LED and OLED control

## Setup and Usage

1. **Hardware Setup**
   - Connect the OLED display to Arduino using I2C
   - Connect Red LED to pin A0
   - Connect Green LED to pin A2
   - Connect Arduino to computer via USB

2. **Software Setup**
   - Upload `sketch_oct15a.ino` to Arduino
   - Create an `image_data` directory in the project root
   - Ensure all Python dependencies are installed

3. **Training the System**
   ```bash
   python CaptureUserImages.py   # Capture user images
   python FaceTrainer.py         # Train the system
   ```

4. **Running the System**
   ```bash
   python FaceRecognize.py
   ```

## System Operation

- Green LED: Indicates successful recognition
- Red LED: Indicates unknown user or no face detected
- OLED Display: Shows status messages
- Face recognition confidence thresholds:
  - Below 40: High confidence match
  - Above 95: Unknown user
  - Between 40-95: Shows confidence percentage

## Troubleshooting

- Ensure proper USB port configuration in `FaceRecognize.py` (ARDUINO_PORT variable)
- Check webcam index if camera isn't detected
- Verify all connections if LEDs or OLED aren't responding
- Ensure proper lighting for optimal face detection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for computer vision tools
- Adafruit for Arduino libraries
- Arduino community for hardware support
