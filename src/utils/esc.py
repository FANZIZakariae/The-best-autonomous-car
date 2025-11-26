# utils/esc.py

import pigpio
import time

ESC_PIN = 2
STOP = 1500
SLOW = 1560

class ESC:
    def __init__(self, pin):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise IOError("pigpio daemon not running! Start with: sudo pigpiod")

        self.pin = pin
        self.current_speed = STOP

        # Arm ESC
        self.pi.set_servo_pulsewidth(self.pin, STOP)
        print("ESC initialized at neutral (1500 µs)")
        time.sleep(2)

    def set_speed(self, us):
        """Just update desired speed"""
        self.current_speed = us

    def update(self):
        """Send one pulse (call this continuously in main loop)"""
        self.pi.set_servo_pulsewidth(self.pin, self.current_speed)

    def stop(self):
        self.current_speed = STOP
        self.update()

    def cleanup(self):
        self.stop()
        self.pi.stop()
