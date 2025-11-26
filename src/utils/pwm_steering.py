import pigpio
import time

class PWMServo:
    """
    Hardware-PWM servo control using pigpio (no jitter).
    """

    def __init__(self, pin=17, min_us=500, max_us=2500):
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("❌ pigpio daemon is not running!")

        print(f"✅ pigpio Servo initialized on GPIO{self.pin}")

    def set_angle(self, angle):
        """Angle: 0° = left, 90° = center, 180° = right"""
        angle = max(0, min(180, angle))
        pulse = self.min_us + (angle / 180.0) * (self.max_us - self.min_us)
        self.pi.set_servo_pulsewidth(self.pin, pulse)

    def turn_left(self):
        self.set_angle(0)

    def turn_right(self):
        self.set_angle(180)

    def center(self):
        self.set_angle(90)

    def cleanup(self):
        self.pi.set_servo_pulsewidth(self.pin, 0)
        self.pi.stop()
        print("🧹 pigpio cleanup done.")
