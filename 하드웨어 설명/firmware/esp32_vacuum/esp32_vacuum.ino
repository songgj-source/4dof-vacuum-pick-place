#include <Arduino.h>

// User wiring: GPIO26 -> L9110S A-1A, GPIO25 -> A-1B.
// Servo UART33/32 is not driven by this pump-only firmware.
constexpr int PUMP_A = 26;
constexpr int PUMP_B = 25;
bool exhausting = false;
unsigned long exhaustStarted = 0;

void stopPump() {
  digitalWrite(PUMP_A, LOW);
  digitalWrite(PUMP_B, LOW);
  exhausting = false;
}

void setup() {
  digitalWrite(PUMP_A, LOW);
  digitalWrite(PUMP_B, LOW);
  pinMode(PUMP_A, OUTPUT);
  pinMode(PUMP_B, OUTPUT);
  stopPump();
  Serial.begin(115200);
  Serial.println("ESP32_VACUUM_L9110S GPIO26/25 ready p/d/s/q");
}

void loop() {
  while (Serial.available()) {
    const char cmd = Serial.read();
    if (cmd == 'p' || cmd == 'P') {
      stopPump();
      digitalWrite(PUMP_A, HIGH);
      Serial.println("[동작] 물건 흡착 (Pick)");
    } else if (cmd == 'd' || cmd == 'D') {
      stopPump();
      digitalWrite(PUMP_B, HIGH);
      exhaustStarted = millis();
      exhausting = true;
      Serial.println("[동작] 역회전 배기 후 물건 놓기 (Drop)");
    } else if (cmd == 's' || cmd == 'S') {
      stopPump();
      Serial.println("[동작] 정지 (Stop)");
    } else if (cmd == 'q' || cmd == 'Q') {
      Serial.println("ESP32_VACUUM_L9110S GPIO26/25");
    }
  }
  if (exhausting && (unsigned long)(millis() - exhaustStarted) >= 300) {
    stopPump();
    Serial.println("[동작] 물건 해제 완료");
  }
}
