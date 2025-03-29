#include <esp_now.h>
#include <WiFi.h>

uint8_t robotMacAddress[] = {0x94, 0x54, 0xC5, 0x74, 0xDC, 0x08};

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("Last Packet Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Computer ESP32 starting...");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_send_cb(OnDataSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, robotMacAddress, 6);
  peerInfo.ifidx = WIFI_IF_STA;
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Computer ESP32 ready to receive from Python and send via ESP-NOW");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.println("Received from Python: " + data);
    esp_err_t result = esp_now_send(robotMacAddress, (uint8_t *)data.c_str(), data.length());
    if (result == ESP_OK) {
      Serial.println("Sent to robot ESP32");
    } else {
      Serial.println("Error sending data");
    }
  }
}