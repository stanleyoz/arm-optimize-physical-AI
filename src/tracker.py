"""
Pan-tilt servo tracker for Physical AI — closed-loop person tracking.

Architecture:
  Pi 5 (perception + control)
    └─ UART/serial ──→ RP2040/R4 (PWM generation)
                           ├─ Pan servo  (GPIO 0)
                           └─ Tilt servo (GPIO 1)

Usage:
  from src.tracker import PanTiltTracker
  tracker = PanTiltTracker(port="/dev/ttyACM0")
  tracker.update(detections, frame_width, frame_height)
  pan, tilt = tracker.current_angles
"""

import time
import serial
from collections import deque


class PIDController:
    def __init__(self, kp=0.5, ki=0.02, kd=0.1, integral_limit=50):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        if dt <= 0:
            return 0.0
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class TargetSelector:
    def __init__(self, target_classes=(0,), max_distance_ratio=0.6):
        self.target_classes = target_classes
        self.max_distance_ratio = max_distance_ratio

    def select(self, detections, frame_center_x, frame_center_y):
        candidates = [d for d in detections if d["class_id"] in self.target_classes]
        if not candidates:
            return None

        best = None
        best_area = 0
        for d in candidates:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = ((cx - frame_center_x) ** 2 + (cy - frame_center_y) ** 2) ** 0.5
            if dist > self.max_distance_ratio * max(frame_center_x, frame_center_y):
                continue
            if area > best_area:
                best_area = area
                best = d
        return best


class PanTiltController:
    def __init__(self, port="/dev/ttyACM0", baud=115200, timeout=0.05):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self.pan = 90
        self.tilt = 90
        self.pan_min, self.pan_max = 0, 180
        self.tilt_min, self.tilt_max = 30, 150
        self._connect()

    def _connect(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(1)
        except serial.SerialException as e:
            print(f"[tracker] Serial not available ({e}). Running in simulation mode.")
            self._ser = None

    def send(self, pan, tilt):
        self.pan = max(self.pan_min, min(self.pan_max, int(pan)))
        self.tilt = max(self.tilt_min, min(self.tilt_max, int(tilt)))
        if self._ser and self._ser.is_open:
            cmd = f"{self.pan},{self.tilt}\n"
            self._ser.write(cmd.encode())

    def center(self):
        self.send(90, 90)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


class PanTiltTracker:
    def __init__(
        self,
        serial_port="/dev/ttyACM0",
        pan_pid=(0.4, 0.01, 0.08),
        tilt_pid=(0.3, 0.01, 0.06),
        target_classes=(0,),
        smoothing=5,
    ):
        self.servo = PanTiltController(port=serial_port)
        self.selector = TargetSelector(target_classes=target_classes)
        self.pan_pid = PIDController(*pan_pid)
        self.tilt_pid = PIDController(*tilt_pid)
        self._prev_time = time.perf_counter()
        self._pan_buffer = deque(maxlen=smoothing)
        self._tilt_buffer = deque(maxlen=smoothing)

    @property
    def current_angles(self):
        return self.servo.pan, self.servo.tilt

    def update(self, detections, frame_width, frame_height):
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now

        cx = frame_width / 2
        cy = frame_height / 2

        target = self.selector.select(detections, cx, cy)
        if target is None:
            self.pan_pid.reset()
            self.tilt_pid.reset()
            self.servo.center()
            return self.current_angles

        x1, y1, x2, y2 = target["bbox"]
        target_cx = (x1 + x2) / 2
        target_cy = (y1 + y2) / 2

        pan_error = target_cx - cx
        tilt_error = target_cy - cy

        pan_delta = self.pan_pid.compute(pan_error, dt if dt > 0 else 0.033)
        tilt_delta = self.tilt_pid.compute(tilt_error, dt if dt > 0 else 0.033)

        new_pan = self.servo.pan - pan_delta
        new_tilt = self.servo.tilt + tilt_delta

        self._pan_buffer.append(new_pan)
        self._tilt_buffer.append(new_tilt)

        smooth_pan = sum(self._pan_buffer) / len(self._pan_buffer)
        smooth_tilt = sum(self._tilt_buffer) / len(self._tilt_buffer)

        self.servo.send(smooth_pan, smooth_tilt)
        return self.current_angles

    def close(self):
        self.servo.center()
        self.servo.close()