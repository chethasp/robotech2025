#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>

// Motor pins (your setup)
const int rightFrontPin = 25;  // D25
const int leftFrontPin = 32;   // D32
const int rightBackPin = 26;   // D26
const int leftBackPin = 33;    // D33

// Timeout settings
const unsigned long timeoutDuration = 5000;  // 10 seconds (adjust as needed)
unsigned long lastDataTime = 0;  // Timestamp of last received data

void OnDataRecv(const esp_now_recv_info_t *recvInfo, const uint8_t *incomingData, int len) {
  char buffer[128];
  if (len < sizeof(buffer)) {
    memcpy(buffer, incomingData, len);
    buffer[len] = '\0';
    Serial.print("Received from computer ESP32: ");
    Serial.println(buffer);

    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, buffer);

    if (error) {
      Serial.print("JSON parsing failed: ");
      Serial.println(error.c_str());
      return;
    }

    // Extract motor values
    int rightFront = doc["rightFront"];
    int leftFront = doc["leftFront"];
    int rightBack = doc["rightBack"];
    int leftBack = doc["leftBack"];

    // Control motors
    digitalWrite(rightFrontPin, rightFront ? HIGH : LOW);
    digitalWrite(leftFrontPin, leftFront ? HIGH : LOW);
    digitalWrite(rightBackPin, rightBack ? HIGH : LOW);
    digitalWrite(leftBackPin, leftBack ? HIGH : LOW);

    // Update last data time
    lastDataTime = millis();

    // Debug output
    Serial.println("Motor states:");
    Serial.print("Right Front: "); Serial.println(rightFront);
    Serial.print("Left Front: "); Serial.println(leftFront);
    Serial.print("Right Back: "); Serial.println(rightBack);
    Serial.print("Left Back: "); Serial.println(leftBack);
  } else {
    Serial.println("Received data too large for buffer");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Robot ESP32 ready to receive ESP-NOW data");

  // Set motor pins as outputs
  pinMode(rightFrontPin, OUTPUT);
  pinMode(leftFrontPin, OUTPUT);
  pinMode(rightBackPin, OUTPUT);
  pinMode(leftBackPin, OUTPUT);

  // Initialize motors to off
  digitalWrite(rightFrontPin, LOW);
  digitalWrite(leftFrontPin, LOW);
  digitalWrite(rightBackPin, LOW);
  digitalWrite(leftBackPin, LOW);

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  if (esp_now_register_recv_cb(OnDataRecv) != ESP_OK) {
    Serial.println("Error registering receive callback");
    return;
  }
}

void loop() {
  // Check if too much time has passed since last data
  if (millis() - lastDataTime > timeoutDuration) {
    // No data received within timeout, default to 0
    digitalWrite(rightFrontPin, LOW);
    digitalWrite(leftFrontPin, LOW);
    digitalWrite(rightBackPin, LOW);
    digitalWrite(leftBackPin, LOW);

    // Optional debug output (comment out if not needed)
    Serial.println("No data received, motors set to 0");
    // delay(1000);  // Slow down debug prints
  }
}