#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ===== WiFi Configuration =====
const char* ssid = "ZTE_2.4G_3chrpy";
const char* password = "E36eeAHv";

// ===== Flask Server URL =====
const char* serverUrl = " http://192.168.1.15:5000";

// ===== Sensor =====
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);

  // ---- WiFi ----
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // ---- I2C ----
  Wire.begin(21, 22);

  // ---- MPU6050 ----
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("MPU6050 initialized");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // ---- JSON Payload ----
    String json =
      "{"
      "\"ax\":" + String(a.acceleration.x, 2) + ","
      "\"ay\":" + String(a.acceleration.y, 2) + ","
      "\"az\":" + String(a.acceleration.z, 2) + ","
      "\"gx\":" + String(g.gyro.x, 2) + ","
      "\"gy\":" + String(g.gyro.y, 2) + ","
      "\"gz\":" + String(g.gyro.z, 2) +
      "}";

    int responseCode = http.POST(json);

    Serial.print("HTTP ");
    Serial.print(responseCode);
    Serial.print(" -> ");
    Serial.println(json);

    http.end();
  }

  // ~20 Hz sampling
  delay(50);
}
