#include <DHT.h>
#include <WiFiNINA.h>

// -------- WiFi --------
const char* WIFI_SSID = "xxx";
const char* WIFI_PASS = "xxx";

// PC server IP (your Flask machine)
IPAddress PC_IP(xxx, xxx, x, xx);
const int PC_PORT = 5000;

// -------- Pin Setup --------
#define DHTPIN    A5
#define DHTTYPE   DHT22

#define SOIL_PIN  A0
#define LIGHT_PIN A3

#define CTRL_PIN  6   // OUTPUT: HIGH=Irrigating YES, LOW=Irrigating NO

DHT dht(DHTPIN, DHTTYPE);

// ---- Calibration Values ----
const int SOIL_DRY = 800;
const int SOIL_WET = 350;

const int LIGHT_DARK = 200;
const int LIGHT_BRIGHT = 900;

const int LIGHT_DETECT_THRESHOLD = 650;

WiFiClient client;

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  int status = WL_IDLE_STATUS;
  while (status != WL_CONNECTED) {
    status = WiFi.begin(WIFI_SSID, WIFI_PASS);
    delay(2000);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("Arduino IP: ");
  Serial.println(WiFi.localIP());
}

bool postJSON(const String& jsonBody) {
  if (!client.connect(PC_IP, PC_PORT)) {
    Serial.println("Failed to connect to PC server for /ingest.");
    return false;
  }

  client.println("POST /ingest HTTP/1.1");
  client.print("Host: ");
  client.print(PC_IP);
  client.print(":");
  client.println(PC_PORT);
  client.println("Content-Type: application/json");
  client.print("Content-Length: ");
  client.println(jsonBody.length());
  client.println("Connection: close");
  client.println();
  client.print(jsonBody);

  // Read & discard response (up to 1.5s)
  unsigned long t0 = millis();
  while (client.connected() && millis() - t0 < 1500) {
    while (client.available()) client.read();
  }

  client.stop();
  return true;
}

// Fetch {"cmd":"START"} or {"cmd":"STOP"} from PC
String getCommand() {
  if (!client.connect(PC_IP, PC_PORT)) {
    Serial.println("Failed to connect to PC server for /cmd.");
    return "";
  }

  client.println("GET /cmd HTTP/1.1");
  client.print("Host: ");
  client.print(PC_IP);
  client.print(":");
  client.println(PC_PORT);
  client.println("Connection: close");
  client.println();

  String resp = "";
  unsigned long t0 = millis();
  while (client.connected() && millis() - t0 < 1500) {
    while (client.available()) {
      char c = client.read();
      resp += c;
    }
  }
  client.stop();

  // Simple parse: look for "cmd":"START" or "cmd":"STOP"
  if (resp.indexOf("\"cmd\":\"START\"") >= 0) return "START";
  if (resp.indexOf("\"cmd\":\"STOP\"") >= 0) return "STOP";
  return "";
}

void applyCommand(const String& cmd) {
  if (cmd == "START") {
    digitalWrite(CTRL_PIN, HIGH);
    Serial.println("CMD=START -> CTRL_PIN HIGH");
  } else if (cmd == "STOP") {
    digitalWrite(CTRL_PIN, LOW);
    Serial.println("CMD=STOP -> CTRL_PIN LOW");
  }
}

bool isIrrigating() {
  // Reading an OUTPUT pin returns the last state written on most Arduino cores.
  return (digitalRead(CTRL_PIN) == HIGH);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("System Initializing...");
  dht.begin();

  pinMode(CTRL_PIN, OUTPUT);
  digitalWrite(CTRL_PIN, LOW);  // default: not irrigating

  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // 1) Poll command and apply it first (so status reflects hardware)
  String cmd = getCommand();
  if (cmd.length() > 0) applyCommand(cmd);

  bool irrigating = isIrrigating();

  // -------- DHT22 --------
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  // -------- Soil --------
  int soilRaw = analogRead(SOIL_PIN);
  int soilPercent = map(soilRaw, SOIL_DRY, SOIL_WET, 0, 100);
  soilPercent = constrain(soilPercent, 0, 100);

  // -------- Light --------
  int lightRaw = analogRead(LIGHT_PIN);
  int lightPercent = map(lightRaw, LIGHT_DARK, LIGHT_BRIGHT, 0, 100);
  lightPercent = constrain(lightPercent, 0, 100);
  bool lightDetected = (lightRaw >= LIGHT_DETECT_THRESHOLD);

  // Print locally
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("DHT22 read failed.");
  } else {
    Serial.print("Temp: "); Serial.print(temperature, 2);
    Serial.print(" °C  | Humidity: "); Serial.print(humidity, 2);
    Serial.println(" %");
  }

  Serial.print("Soil Raw Value: "); Serial.print(soilRaw);
  Serial.print("  | Soil Moisture: "); Serial.print(soilPercent); Serial.println(" %");

  Serial.print("Light Raw Value: "); Serial.print(lightRaw);
  Serial.print("  | Light Level: "); Serial.print(lightPercent);
  Serial.print(" %  | Light Detected: "); Serial.println(lightDetected ? "YES" : "NO");

  Serial.print("Irrigating: ");
  Serial.println(irrigating ? "YES" : "NO");
  Serial.println("-----------------------------");

  // Build JSON payload (includes irrigating status from real pin state)
  String json = "{";
  json += "\"temp_c\":" + String(temperature, 2) + ",";
  json += "\"humidity\":" + String(humidity, 2) + ",";
  json += "\"soil_raw\":" + String(soilRaw) + ",";
  json += "\"soil_percent\":" + String(soilPercent) + ",";
  json += "\"light_raw\":" + String(lightRaw) + ",";
  json += "\"light_percent\":" + String(lightPercent) + ",";
  json += "\"light_detected\":\"" + String(lightDetected ? "YES" : "NO") + "\",";
  json += "\"irrigating\":\"" + String(irrigating ? "YES" : "NO") + "\"";
  json += "}";

  if (postJSON(json)) {
    Serial.println("Uploaded to PC ✅");
  } else {
    Serial.println("Upload failed ❌");
  }

  delay(2000);
}




