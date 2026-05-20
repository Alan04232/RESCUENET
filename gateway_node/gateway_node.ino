#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <HTTPClient.h>

// ================= WiFi =================
const char* ssid = "N00384";
const char* password = "01010123";
const char* serverURL = "http://10.71.73.5:5000/node-data";

// ================= Globals =================
volatile bool newPacket = false;
char incomingJSON[250];   // fixed buffer (more stable than String)

// ================= ESP-NOW RECEIVE =================
void onReceive(const esp_now_recv_info *info, const uint8_t *data, int len) {

  Serial.println("\n===== JSON RECEIVED FROM NODE =====");

  Serial.print("Sender MAC: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", info->src_addr[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.println();

  // Copy data safely and remove possible NULL terminator
  int copyLen = len;
  if (copyLen >= sizeof(incomingJSON))
      copyLen = sizeof(incomingJSON) - 1;

  memcpy(incomingJSON, data, copyLen);
  incomingJSON[copyLen] = '\0';   // ensure proper string termination

  Serial.println("Raw JSON:");
  Serial.println(incomingJSON);
  Serial.println("==============================");

  newPacket = true;
}

// ================= WIFI CONNECT =================
void connectWiFi() {

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println(" connected");
  Serial.print("Gateway IP: ");
  Serial.println(WiFi.localIP());

  WiFi.setSleep(false);

  // Lock ESP-NOW to same channel
  esp_wifi_set_channel(WiFi.channel(), WIFI_SECOND_CHAN_NONE);
}

// ================= SETUP =================
void setup() {

  Serial.begin(115200);

  connectWiFi();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_register_recv_cb(onReceive);

  Serial.println("Gateway ready (ESP-NOW + WiFi)");
}

// ================= LOOP =================
void loop() {

  if (!newPacket) return;

  newPacket = false;

  Serial.println("\n----- JSON SENT TO SERVER -----");
  Serial.println(incomingJSON);

  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    // Send exact byte length (NO extra NULL)
    int code = http.POST(
      (uint8_t*)incomingJSON,
      strlen(incomingJSON)
    );

    Serial.print("Server response: ");
    Serial.println(code);

    http.end();

  } else {
    Serial.println("WiFi lost");
  }

  Serial.println("------------------------------");
}