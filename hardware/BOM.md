# Physical AI Pan-Tilt Tracker — Bill of Materials

## Required Components

| Item | Qty | Approx. Cost | Notes |
|------|-----|-------------|-------|
| Raspberry Pi 5 (4+ GB) | 1 | $60–80 | Runs inference + control loop |
| Raspberry Pi Camera Module 3 | 1 | $25 | Or any USB webcam |
| **SR431 Servo** (pan) | 1 | $8–12 | Standard analog servo, 0–180° |
| **SR431 Servo** (tilt) | 1 | $8–12 | Standard analog servo, 0–180° |
| Raspberry Pi Pico / RP2040 | 1 | $4–8 | Dedicated PWM co-processor |
| Servo shield (or mini breadboard) | 1 | $5–10 | Power distribution for servos |
| 5V / 2A external power supply | 1 | $8–12 | For servos (do NOT power from Pi 5V) |
| Pan-tilt camera bracket (printed) | 1 | <$1 | STL in hardware/ bracket.stl |
| USB cable (Micro B) | 1 | $3 | RP2040 → Pi 5 (serial communication) |
| Jumper wires (M-F) | 10 | $2 | Wiring |
| 3D printer filament (PLA) | ~20g | <$1 | Optional — bracket printing |

## Total Cost (excl. Pi 5): ~$30–40

## Power Budget

| Component | Current | Notes |
|-----------|---------|-------|
| Pi 5 (idle + camera) | ~2.5 A @ 5V | Via official Pi 5 PSU |
| 2× SR431 servos (idle) | ~10 mA each | Holding position |
| 2× SR431 servos (moving) | ~200–500 mA each | Peak under load |
| RP2040 | ~20 mA | via USB from Pi 5 |

**Recommendation:** Use a separate 5V / 2A regulated supply for the servos.
Common ground with Pi 5 is essential.

## Wiring Overview

```
Pi 5 (USB)  ──USB── RP2040 ──GP0── Pan SR431 (signal)
                            ──GP1── Tilt SR431 (signal)
                            ──VBUS── Servo VCC (5V ext. PSU)
                            ──GND─── Servo GND (common ground)
                                      Pi 5 GND (common ground)
```