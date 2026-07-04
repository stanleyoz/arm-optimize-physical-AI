# Hardware Assembly Guide — Pan-Tilt Physical AI Tracker

## Step 1: Assemble the Pan-Tilt Bracket

1. Print the pan-tilt bracket (STL files in `hardware/bracket/`)
2. Mount the Pi Camera Module 3 to the tilt platform
3. Secure the pan SR431 servo into the base
4. Secure the tilt SR431 servo to the moving arm
5. Attach tilt platform to servo horn

## Step 2: Wire the Servos to the RP2040

```
SR431 (Pan)  →  RP2040 GP0  (yellow/white signal wire)
SR431 (Tilt) →  RP2040 GP1  (yellow/white signal wire)
Both servos  →  RP2040 VBUS (red, 5V external PSU — NOT from RP2040 3V3)
Both servos  →  RP2040 GND  (brown/black, shared ground)
```

**IMPORTANT:** Power the servos from an external 5V supply, not the RP2040's 3.3V pin.
Connect the external supply ground to RP2040 GND.

## Step 3: Connect RP2040 to Pi 5

- USB cable: Pi 5 USB port → RP2040 micro USB
- The RP2040 appears as `/dev/ttyACM0` on the Pi 5

## Step 4: Flash the RP2040 Firmware

1. Install Arduino IDE or PlatformIO on the Pi 5
2. Add RP2040 board support (Raspberry Pi Pico)
3. Open `hardware/pan_tilt_firmware/pan_tilt.ino`
4. Select board: "Raspberry Pi Pico"
5. Compile and upload

## Step 5: Test the Servos

```bash
# From the Pi 5, with RP2040 connected:
python -c "
import serial, time
ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(1)
ser.write(b'90,90\\n')   # center
time.sleep(1)
ser.write(b'45,90\\n')   # pan left
time.sleep(1)
ser.write(b'135,90\\n')  # pan right
time.sleep(1)
ser.write(b'90,60\\n')   # tilt up
time.sleep(1)
ser.write(b'90,120\\n')  # tilt down
ser.close()
print('Servos OK')
"
```

## Step 6: Connect Pi Camera

- Connect Pi Camera Module 3 ribbon cable to Pi 5 CAM connector
- Enable camera: `sudo raspi-config` → Interface Options → Camera → Enable
- Verify: `libcamera-hello --timeout 5000`

## Step 7: Run the Tracker

```bash
python run.py --track
```