#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define MPU6050_ADDR 0x68

// MPU6050 Registers
#define MPU6050_REG_PWR_MGMT_1 0x6B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B
#define MPU6050_REG_TEMP_OUT_H 0x41
#define MPU6050_REG_GYRO_XOUT_H 0x43

// Soil moisture sensor
#define SOIL_PIN 34
#define SOIL_DRY 4095
#define SOIL_WET 1500

// Vibration thresholds
#define VIBRATION_THRESHOLD 2.0  // g (acceleration due to gravity)
#define SAMPLES_FOR_AVERAGE 10

// Structure to hold sensor data
typedef struct {
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
  float temperature;
  int soilMoisture;
  float vibrationMagnitude;
  bool isVibrating;
} SensorData;

SensorData sensorData;
int vibrationSampleCount = 0;
float vibrationSum = 0;

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n╔════════════════════════════════════════╗");
  Serial.println("║   SOIL VIBRATION MONITORING SYSTEM    ║");
  Serial.println("║         MPU6050 + Soil Sensor        ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  delay(500);
  
  // Initialize MPU6050
  initMPU6050();
  
  // Initialize analog reading
  analogReadResolution(12);
  
  Serial.println("✓ System initialized and ready\n");
  Serial.println("═══════════════════════════════════════════\n");
  
  delay(1000);
}

void loop() {
  // Read all sensors
  readAccelerometer();
  readGyroscope();
  readTemperature();
  readSoilMoisture();
  
  // Calculate vibration magnitude
  calculateVibrationMagnitude();
  
  // Print data
  printSensorData();
  
  // Check for vibration
  if (sensorData.isVibrating) {
    Serial.println("  🔴 VIBRATION DETECTED!");
  }
  
  Serial.println("───────────────────────────────────────────\n");
  
  delay(1000);  // Read every 1 second
}

// ===== MPU6050 INITIALIZATION =====
void initMPU6050() {
  Serial.println("Initializing MPU6050...");
  
  // Wake up sensor (clear sleep bit)
  byte pwr = readRegister(MPU6050_REG_PWR_MGMT_1);
  writeRegister(MPU6050_REG_PWR_MGMT_1, pwr & 0xBF);
  delay(100);
  
  // Set accelerometer range to ±2g
  writeRegister(MPU6050_REG_ACCEL_CONFIG, 0x00);
  
  Serial.println("✓ MPU6050 ready\n");
}

// ===== READ ACCELEROMETER =====
void readAccelerometer() {
  int16_t rawX = readRegister16(MPU6050_REG_ACCEL_XOUT_H);
  int16_t rawY = readRegister16(MPU6050_REG_ACCEL_XOUT_H + 2);
  int16_t rawZ = readRegister16(MPU6050_REG_ACCEL_XOUT_H + 4);
  
  // Convert to g (16384 LSB/g for ±2g range)
  sensorData.accelX = rawX / 16384.0;
  sensorData.accelY = rawY / 16384.0;
  sensorData.accelZ = rawZ / 16384.0;
}

// ===== READ GYROSCOPE =====
void readGyroscope() {
  int16_t rawX = readRegister16(MPU6050_REG_GYRO_XOUT_H);
  int16_t rawY = readRegister16(MPU6050_REG_GYRO_XOUT_H + 2);
  int16_t rawZ = readRegister16(MPU6050_REG_GYRO_XOUT_H + 4);
  
  // Convert to degrees/sec (131 LSB/°/s for ±250°/s range)
  sensorData.gyroX = rawX / 131.0;
  sensorData.gyroY = rawY / 131.0;
  sensorData.gyroZ = rawZ / 131.0;
}

// ===== READ TEMPERATURE =====
void readTemperature() {
  int16_t rawTemp = readRegister16(MPU6050_REG_TEMP_OUT_H);
  
  // Convert to Celsius (333.87 LSB/°C, 21°C at 0)
  sensorData.temperature = (rawTemp / 333.87) + 21.0;
}

// ===== READ SOIL MOISTURE =====
void readSoilMoisture() {
  int raw = analogRead(SOIL_PIN);
  
  // Map from ADC range to percentage
  int moisture = map(raw, SOIL_DRY, SOIL_WET, 0, 100);
  sensorData.soilMoisture = constrain(moisture, 0, 100);
}

// ===== CALCULATE VIBRATION MAGNITUDE =====
void calculateVibrationMagnitude() {
  // Remove gravity component (assume Z is vertical with ~1g)
  float ax = sensorData.accelX;
  float ay = sensorData.accelY;
  float az = sensorData.accelZ - 1.0;  // Remove gravity
  
  // Calculate magnitude
  float magnitude = sqrt(ax*ax + ay*ay + az*az);
  
  // Average over multiple samples
  vibrationSum += magnitude;
  vibrationSampleCount++;
  
  if (vibrationSampleCount >= SAMPLES_FOR_AVERAGE) {
    sensorData.vibrationMagnitude = vibrationSum / SAMPLES_FOR_AVERAGE;
    sensorData.isVibrating = (sensorData.vibrationMagnitude > VIBRATION_THRESHOLD);
    
    vibrationSum = 0;
    vibrationSampleCount = 0;
  }
}

// ===== PRINT SENSOR DATA =====
void printSensorData() {
  Serial.println("┌─ ACCELEROMETER ─────────────────────────┐");
  Serial.print("│ X: "); Serial.print(sensorData.accelX, 3); Serial.print(" g  ");
  Serial.print("Y: "); Serial.print(sensorData.accelY, 3); Serial.print(" g  ");
  Serial.print("Z: "); Serial.print(sensorData.accelZ, 3); Serial.println(" g │");
  
  Serial.println("├─ GYROSCOPE ─────────────────────────────┤");
  Serial.print("│ X: "); Serial.print(sensorData.gyroX, 2); Serial.print(" °/s  ");
  Serial.print("Y: "); Serial.print(sensorData.gyroY, 2); Serial.print(" °/s  ");
  Serial.print("Z: "); Serial.print(sensorData.gyroZ, 2); Serial.println(" °/s │");
  
  Serial.println("├─ ENVIRONMENT ──────────────────────────┤");
  Serial.print("│ Temperature: "); Serial.print(sensorData.temperature, 1); Serial.println(" °C        │");
  Serial.print("│ Soil Moisture: "); Serial.print(sensorData.soilMoisture); Serial.println(" %       │");
  
  Serial.println("├─ VIBRATION ANALYSIS ───────────────────┤");
  Serial.print("│ Magnitude: "); Serial.print(sensorData.vibrationMagnitude, 3); Serial.println(" g      │");
  Serial.print("│ Threshold: "); Serial.print(VIBRATION_THRESHOLD, 1); Serial.println(" g          │");
  Serial.print("│ Status: ");
  if (sensorData.isVibrating) {
    Serial.println("VIBRATING        │");
  } else {
    Serial.println("STABLE           │");
  }
  Serial.println("└─────────────────────────────────────────┘");
}

// ===== I2C HELPER FUNCTIONS =====
byte readRegister(byte reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 1);
  return Wire.read();
}

void writeRegister(byte reg, byte value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

int16_t readRegister16(byte reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 2);
  
  byte high = Wire.read();
  byte low = Wire.read();
  
  return (int16_t)((high << 8) | low);
}