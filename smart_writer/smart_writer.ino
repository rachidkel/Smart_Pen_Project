#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <MPU6050_light.h>

// ================= WiFi =================
const char* ssid = "ZTE_2.4G_3chrpy";
const char* password = "E36eeAHv";

// ================= Server =================
const char* serverURL = "http://192.168.1.11:5000/data";

// ================= MPU =================
MPU6050 mpu(Wire);

// ================= Timing =================
unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 10; // ms → 100 Hz

// ================= WiFi Connect =================
void connectToWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(300);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ WiFi Failed");
  }
}

void setup() {
  Serial.begin(9600);
  Wire.begin(21, 22); // ESP32 I2C

  connectToWiFi();

  byte status = mpu.begin();
  if (status != 0) {
    Serial.println("❌ MPU6050 not found!");
    while (1) delay(10);
  }

  Serial.println("Calibrating MPU6050...");
  delay(1000);
  mpu.calcOffsets();   // keep still
  Serial.println("✅ MPU6050 Ready");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    return;
  }

  unsigned long now = millis();
  if (now - lastSend < SEND_INTERVAL) return;
  lastSend = now;

  mpu.update();

  float ax = mpu.getAccX();
  float ay = mpu.getAccY();
  float az = mpu.getAccZ();
  float gx = mpu.getGyroX();
  float gy = mpu.getGyroY();
  float gz = mpu.getGyroZ();

  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "application/json");

  char payload[128];
  snprintf(payload, sizeof(payload),
           "{\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.3f,\"gy\":%.3f,\"gz\":%.3f}",
           ax, ay, az, gx, gy, gz);

  int httpCode = http.POST((uint8_t*)payload, strlen(payload));
  http.end();

  Serial.printf("POST %d | %s\n", httpCode, payload);
}
