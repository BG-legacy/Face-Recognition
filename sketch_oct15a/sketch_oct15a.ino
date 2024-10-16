#include <Wire.h>               // Include Wire library for I2C communication
#include <Adafruit_GFX.h>       // Include Adafruit GFX library for graphics
#include <Adafruit_SSD1306.h>   // Include Adafruit SSD1306 library for OLED display

// Define the OLED display dimensions
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// Define the OLED reset pin. -1 means sharing Arduino reset pin
#define OLED_RESET -1
// Create an OLED display object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Define LED pin numbers
int redLED = A0;   // Red LED connected to analog pin A0
int greenLED = A2; // Green LED connected to analog pin A2

void setup() {
  // Initialize Serial communication at 9600 baud rate
  Serial.begin(9600);

  // Initialize LEDs as output pins
  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);

  // Turn off all LEDs initially
  digitalWrite(redLED, LOW);
  digitalWrite(greenLED, LOW);

  // Initialize the OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;); // Stay in infinite loop if initialization fails
  }
  // Clear the display buffer and update the display
  display.clearDisplay();
  display.display();
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    // Read the incoming message until newline character
    String message = Serial.readStringUntil('\n');

    // Display the message on the OLED
    displayMessage(message);

    // Control LEDs based on the received message
    if (message == "User not identified") {
      digitalWrite(redLED, HIGH);   // Turn on Red LED
      digitalWrite(greenLED, LOW);  // Turn off Green LED
    } else if (message == "Welcome Home Boss") {
      digitalWrite(greenLED, HIGH); // Turn on Green LED
      digitalWrite(redLED, LOW);    // Turn off Red LED
    }
  }
}

// Function to display message on the OLED
void displayMessage(String msg) {
  display.clearDisplay();           // Clear the display buffer
  display.setTextSize(2);           // Set text size to 2
  display.setTextColor(SSD1306_WHITE); // Set text color to white
  display.setCursor(0, 20);         // Set cursor position
  display.println(msg);             // Print the message
  display.display();                // Update the display
}

// Resources for further learning:
// 1. Arduino Serial Communication: https://www.arduino.cc/reference/en/language/functions/communication/serial/
// 2. Adafruit SSD1306 Library: https://github.com/adafruit/Adafruit_SSD1306
// 3. Adafruit GFX Library: https://learn.adafruit.com/adafruit-gfx-graphics-library
// 4. Arduino Digital I/O: https://www.arduino.cc/reference/en/language/functions/digital-io/digitalwrite/
// 5. I2C Communication: https://www.arduino.cc/en/reference/wire
// 6. OLED Display Tutorial: https://randomnerdtutorials.com/guide-for-oled-display-with-arduino/
// 7. LED Control with Arduino: https://www.arduino.cc/en/Tutorial/BuiltInExamples/Blink
// 8. Arduino String Handling: https://www.arduino.cc/reference/en/language/variables/data-types/stringobject/
