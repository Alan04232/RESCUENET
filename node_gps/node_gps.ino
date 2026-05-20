/*
 * INTELLIGENT MULTI-HAZARD DISASTER EARLY WARNING SYSTEM
 * SENSOR NODE 1 - JSON PAYLOAD VERSION
 * Manual MPU Driver (MPU6500 / MPU6050 Compatible)
 */

#include <WiFi.h>
#include <esp_now.h>
#include <Wire.h>
#include <esp_wifi.h>
#include <string.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

// ========== PIN DEFINITIONS ==========
#define SOIL_PIN 34
#define SDA_PIN 21
#define SCL_PIN 22
#define FLAME_DO 27
#define FLAME_AO 35

// ========== MPU REGISTER DEFINITIONS ==========
#define MPU_ADDR 0x68
#define MPU_REG_PWR_MGMT_1   0x6B
#define MPU_REG_ACCEL_CONFIG 0x1C
#define MPU_REG_GYRO_CONFIG  0x1B
#define MPU_REG_CONFIG       0x1A
#define MPU_REG_ACCEL_XOUT_H 0x3B
#define MPU_REG_TEMP_OUT_H   0x41

// ========== NODE CONFIG ==========
#define NODE_ID 2
#define SENSOR_READ_INTERVAL_MS 500
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);
float LAT = 0.0;
float LON = 0.0;

uint8_t GATEWAY_MAC[] = {0x00, 0x4B, 0x12, 0x3D, 0x28, 0x24};

bool mpuOK = false;
uint8_t mpuErrorCount = 0;
// ========== ESP-NOW CALLBACK ==========
void onSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS)
    Serial.println("✓ Packet sent successfully");
  else
    Serial.println("✗ Packet send failed");
}

// ========== I2C HELPERS ==========
void writeRegister(byte reg, byte value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

int16_t readRegister16(byte reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 2);

  if (Wire.available() < 2) {
    mpuErrorCount++;
    return 0;
  }

  byte high = Wire.read();
  byte low  = Wire.read();
  return (int16_t)((high << 8) | low);
}

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);   // ✅ ADD THIS FIRST
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);

  pinMode(FLAME_DO, INPUT);
  pinMode(SOIL_PIN, INPUT);
  analogReadResolution(12);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(false);
  esp_wifi_set_channel(11, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Init Failed!");
    while (1);
  }
  esp_now_register_send_cb(onSent);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, GATEWAY_MAC, 6);
  peer.channel = 11;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Peer Add Failed!");
    while (1);
  }
  // I2C Init
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  delay(200);
  // MPU Init
  writeRegister(MPU_REG_PWR_MGMT_1, 0x00);
  delay(100);
  writeRegister(MPU_REG_ACCEL_CONFIG, 0x00);  // ±2g
  writeRegister(MPU_REG_GYRO_CONFIG, 0x00);   // ±250 dps
  writeRegister(MPU_REG_CONFIG, 0x04);        // ~20Hz LPF
  delay(100);

  int16_t test = readRegister16(MPU_REG_ACCEL_XOUT_H);
  if (test != 0 || mpuErrorCount == 0) {
    mpuOK = true;
    Serial.println("MPU Initialized ✓");
  } else {
    mpuOK = false;
    Serial.println("MPU Failed ✗");
  }
  Serial.println("Node Ready");
}

// ========== LOOP ==========
void loop() {
  while (gpsSerial.available()) {
  gps.encode(gpsSerial.read());
  }
  if (gps.location.isValid()) {
  LAT = gps.location.lat();
  LON = gps.location.lng();
  } else {
  Serial.println("Waiting for GPS fix...");
  }
  uint32_t timestamp = millis() / 1000;
  int soil_raw = analogRead(SOIL_PIN);
  float vib_x = 0;
  float vib_y = 0;
  float vib_z = 0;
  float temperature = 0;

  if (mpuOK) {
    int16_t rawX = readRegister16(MPU_REG_ACCEL_XOUT_H);
    int16_t rawY = readRegister16(MPU_REG_ACCEL_XOUT_H + 2);
    int16_t rawZ = readRegister16(MPU_REG_ACCEL_XOUT_H + 4);
    int16_t rawTemp = readRegister16(MPU_REG_TEMP_OUT_H);
    vib_x = (rawX / 16384.0) * 9.80665;
    vib_y = (rawY / 16384.0) * 9.80665;
    vib_z = (rawZ / 16384.0) * 9.80665;
    temperature = (rawTemp / 333.87) + 21.0;
  }

  int flame_adc = analogRead(FLAME_AO);
  bool flame = (digitalRead(FLAME_DO) == LOW);

  // ===== JSON PAYLOAD =====
  char payload[240];
  snprintf(payload, sizeof(payload),
  "{\"node_id\":%d,"
  "\"lat\":%.5f,"
  "\"lon\":%.5f,"
  "\"soil_moisture\":%d,"
  "\"vib_x\":%.3f,"
  "\"vib_y\":%.3f,"
  "\"vib_z\":%.3f,"
  "\"humidity\":%.1f,"
  "\"distance\":%.1f,"
  "\"flame\":%s}",
  NODE_ID,
  LAT,
  LON,
  soil_raw,
  vib_x,
  vib_y,
  vib_z,
  temperature,
  (float)flame_adc,
  flame ? "true" : "false"
  );
  Serial.println(payload);
  Serial.print("Payload Size: ");
  Serial.println(strlen(payload));
  // Send JSON
  esp_err_t result = esp_now_send(
    GATEWAY_MAC,
    (uint8_t*)payload,
    strlen(payload) + 1
  );
  if (result != ESP_OK) {
    Serial.print("Send Error: ");
    Serial.println(result);
  }
  delay(SENSOR_READ_INTERVAL_MS);
}