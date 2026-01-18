import pigpio
import time

ESC_PIN = 2

class ESC:
    def __init__(self, pin):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise IOError("pigpio daemon not running! Start with: sudo pigpiod")

        self.pin = pin
        
        # Arm ESC at Neutral
        print("Arming ESC at neutral (1500 us)...")
        self.pi.set_servo_pulsewidth(self.pin, 1500)
        time.sleep(2) # Wait for ESC to initialize

    def set_speed(self, us):
        """Set raw PWM pulse"""
        # Safety clamp to prevent errors (though 0 is allowed for off)
        if us < 0: us = 0
        self.pi.set_servo_pulsewidth(self.pin, us)

    def cleanup(self):
        """Stop and cleanup"""
        self.set_speed(0)
        self.pi.stop()

# ---------------------------------------------------
# BACKWARD RAMP TEST SEQUENCE
# ---------------------------------------------------
if __name__ == "__main__":
    esc = ESC(ESC_PIN)

    try:
        # Loop from 1500 down to 0, stepping down by 100
        for pulse in range(1500, -100, -100):
            print(f"Testing Pulse: {pulse} us")
            esc.set_speed(pulse)
            
            # Wait 1 second to observe movement
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    finally:
        print("Cleaning up...")
        esc.cleanup()
        print("Done.")
