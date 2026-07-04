/**
 * Pan-Tilt Servo Controller for Raspberry Pi 5 Physical AI Tracker
 *
 * Receives angle commands over UART from Pi 5 and generates PWM signals
 * for two SR431 servos (pan + tilt).
 *
 * Protocol:  "PAN,TILT\\n"   (e.g. "90,90\\n" = center)
 * Range:     0–180 degrees per servo
 * Default:   90,90 (center)
 *
 * Pinout:
 *   GP0  → Pan servo  (signal wire)
 *   GP1  → Tilt servo (signal wire)
 *   UART → Serial via USB (to Pi 5) or GP16/GP17 (UART0)
 */

#include <Arduino.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN = 0;
const int TILT_PIN = 1;

int panAngle = 90;
int tiltAngle = 90;

String inputBuffer = "";

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    panServo.attach(PAN_PIN);
    tiltServo.attach(TILT_PIN);

    panServo.write(panAngle);
    tiltServo.write(tiltAngle);

    Serial.println("SR431 Pan-Tilt Ready");
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\\n') {
            int comma = inputBuffer.indexOf(',');
            if (comma > 0) {
                int p = inputBuffer.substring(0, comma).toInt();
                int t = inputBuffer.substring(comma + 1).toInt();
                panAngle = constrain(p, 0, 180);
                tiltAngle = constrain(t, 30, 150);
                panServo.write(panAngle);
                tiltServo.write(tiltAngle);
            }
            inputBuffer = "";
        } else if (c != '\\r') {
            inputBuffer += c;
        }
    }
}