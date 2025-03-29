#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>

// Motor pins (adjust these to your actual setup)
const int rightFrontPin = 2;  // D2
const int leftFrontPin = 3;   // D3
const int rightBackPin = 4;   // D4
const int leftBackPin = 5;    // D5

void OnDataRecv(const esp_now_recv_info_t *recvInfo, const uint8_t *incomingData, int len) {
  char buffer[128];
  if (len < sizeof(buffer)) {
    memcpy(buffer, incomingData, len);
    buffer[len] = '\0';
    Serial.print("Received from computer ESP32: ");
    Serial.println(buffer);

    // Parse JSON
    StaticJsonDocument<200> doc;  // Buffer size for JSON
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
  // Nothing here
}