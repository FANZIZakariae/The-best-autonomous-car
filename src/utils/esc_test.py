# utils/esc.py

import pigpio
import time

ESC_PIN = 2

# Standard ESC pulse widths
STOP = 1500
FULL_FORWARD = 1600
FULL_REVERSE = 1400


class ESC:
    def __init__(self, pin):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise IOError("pigpio daemon not running! Start with: sudo pigpiod")

        self.pin = pin
        self.current_speed = STOP

        # Arm ESC
        self.pi.set_servo_pulsewidth(self.pin, STOP)
        print("ESC initialized at neutral (1500 s)")
        time.sleep(2)

    def set_speed(self, us):
        """Set raw PWM pulse"""
        self.current_speed = us
        self.pi.set_servo_pulsewidth(self.pin, self.current_speed)

    def forward(self, speed_us=FULL_FORWARD):
        """Move forward"""
        self.set_speed(speed_us)

    def backward(self, speed_us=FULL_REVERSE):
        """Move backward"""
        self.set_speed(speed_us)

    def stop(self):
        """Stop motor"""
        self.set_speed(STOP)

    def cleanup(self):
        """Stop and cleanup"""
        self.stop()
        self.pi.set_servo_pulsewidth(self.pin, 0)
        self.pi.stop()


# ---------------------------------------------------
# SIMPLE TEST SEQUENCE
# ---------------------------------------------------
if __name__ == "__main__":
    esc = ESC(ESC_PIN)

    try:
        print("? Backward for 1 second")
        esc.backward()
        time.sleep(5)
        
        print(" Stop for 1 second")
        esc.stop()
        time.sleep(2)
        
        print("? Forward for 1 second")
        esc.forward()
        time.sleep(5)

        print(" Stop for 1 second")
        esc.stop()
        time.sleep(2)

        print("? Backward for 1 second")
        esc.backward()
        time.sleep(5)

        print(" Final stop")
        esc.stop()
        time.sleep(1)

    finally:
        esc.cleanup()
        print("ESC test finished & cleaned up")
