#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_AS7343.h>

// ── WiFi Hotspot Config ───────────────────────────────────────────────────────
#define AP_SSID     "CAPSTONE_AP"
#define AP_PASS     "capstone123"
#define UDP_PORT    4210
#define UDP_DEST    "192.168.4.255"  // broadcast to all clients on AP

// ── Sensor Config ─────────────────────────────────────────────────────────────
#define MUX_ADDR    0x70
#define NUM_SENSORS 8   // channels 0-7

Adafruit_AS7343 as7343[NUM_SENSORS];
bool sensor_ok[NUM_SENSORS] = {false};

WiFiUDP udp;

// ── helpers ───────────────────────────────────────────────────────────────────

void selectMuxChannel(uint8_t ch) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

void clearMux() {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// ── setup ─────────────────────────────────────────────────────────────────────

bool setupSensor(uint8_t idx) {
  selectMuxChannel(idx);
  delay(10);
  if (!i2cPresent(0x39)) {
    Serial.printf("ERROR: AS7343[%d] not found\n", idx);
    clearMux(); return false;
  }
  if (!as7343[idx].begin()) {
    Serial.printf("ERROR: AS7343[%d] begin() failed\n", idx);
    clearMux(); return false;
  }
  as7343[idx].setGain(AS7343_GAIN_256X);
  as7343[idx].setATIME(2);
  as7343[idx].setASTEP(599);
  as7343[idx].setLEDCurrent(4);
  as7343[idx].enableLED(true);
  clearMux();
  Serial.printf("AS7343[%d] OK\n", idx);
  return true;
}

// ── send via UDP ──────────────────────────────────────────────────────────────

void sendSensor(uint8_t idx, uint16_t *r) {
  char buf[120];
  snprintf(buf, sizeof(buf), "S%d:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
    idx,
    r[AS7343_CHANNEL_F1],  r[AS7343_CHANNEL_F2],
    r[AS7343_CHANNEL_FZ],  r[AS7343_CHANNEL_F3],
    r[AS7343_CHANNEL_F4],  r[AS7343_CHANNEL_F5],
    r[AS7343_CHANNEL_FY],  r[AS7343_CHANNEL_FXL],
    r[AS7343_CHANNEL_F6],  r[AS7343_CHANNEL_F7],
    r[AS7343_CHANNEL_F8],  r[AS7343_CHANNEL_NIR],
    r[AS7343_CHANNEL_VIS_TL_0]);

  udp.beginPacket(UDP_DEST, UDP_PORT);
  udp.print(buf);
  udp.endPacket();
}

// ── main ──────────────────────────────────────────────────────────────────────

void setup() {
  delay(2000);
  Serial.begin(115200);
  delay(500);
  Serial.println("\n8x AS7343 WiFi starting...");

  pinMode(7, OUTPUT);
  digitalWrite(7, HIGH);
  delay(20);

  Wire.begin();
  Wire.setClock(1000000);
  delay(50);

  // Start WiFi hotspot
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.printf("Hotspot: %s  |  IP: %s\n", AP_SSID,
                WiFi.softAPIP().toString().c_str());

  udp.begin(UDP_PORT);
  Serial.printf("UDP broadcasting on port %d\n", UDP_PORT);

  if (!i2cPresent(MUX_ADDR)) {
    Serial.println("ERROR: mux not found - halting");
    while (true) delay(1000);
  }

  for (uint8_t i = 0; i < NUM_SENSORS; i++)
    sensor_ok[i] = setupSensor(i);

  Serial.println("Ready — connect laptop to CAPSTONE_AP and run wireless_live_plot.py");
}

void loop() {
  // Step 1: trigger all simultaneously
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    if (!sensor_ok[i]) continue;
    selectMuxChannel(i);
    as7343[i].startMeasurement();
    clearMux();
  }

  // Step 2: wait for all to finish
  bool done[NUM_SENSORS] = {false};
  uint32_t t = millis();
  while (true) {
    bool all_done = true;
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
      if (!sensor_ok[i]) { done[i] = true; continue; }
      if (done[i]) continue;
      selectMuxChannel(i);
      if (as7343[i].dataReady()) done[i] = true;
      else all_done = false;
      clearMux();
    }
    if (all_done) break;
    if (millis() - t > 500) { Serial.println("ERROR: timeout"); break; }
  }

  // Step 3: read and send via UDP
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    if (!sensor_ok[i] || !done[i]) continue;
    selectMuxChannel(i);
    uint16_t readings[18] = {0};
    if (as7343[i].readAllChannels(readings)) sendSensor(i, readings);
    else Serial.printf("ERROR: AS7343[%d] read failed\n", i);
    clearMux();
  }
}
