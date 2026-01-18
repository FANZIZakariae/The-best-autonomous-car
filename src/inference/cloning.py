import json
import time
import paho.mqtt.client as mqtt
from utils.pwm_steering import PWMServo
from utils.esc import ESC

# --- Utility mapping functions ---

def map_angle(mqtt_angle):
    """
    Maps MQTT angle [-1, 1] to servo angle [0, 180]
    """
    mqtt_angle = max(-1.0, min(1.0, mqtt_angle))
    return (mqtt_angle + 1.0) * 90.0


def map_throttle(mqtt_throttle):
    """
    Maps MQTT throttle [-1, 1] ? ESC range
    For now, forward only. Reverse not implemented.
    """
    return max(0.0, min(1.0, mqtt_throttle))


# --- Initialize actuators ---

servo = PWMServo(pin=17)
esc = ESC(pin=2)

# Start motor at stopped / neutral
esc.set_speed(1560)  # adjust if your ESC uses different neutral value

# Optional smoothing for servo to avoid snapping
smoothed_angle = 90
alpha = 0.2  # smoothing factor


# --- MQTT callback ---

def on_message(client, userdata, msg):
    global smoothed_angle

    try:
        data = json.loads(msg.payload.decode())
        raw_angle = data["angle"]
        raw_throttle = data["throttle"]

        # Map to actuator values
        servo_angle = map_angle(raw_angle)
        throttle_val = map_throttle(raw_throttle)

        # Smooth steering
        smoothed_angle = smoothed_angle * (1 - alpha) + servo_angle * alpha

        # Send to actuators
        servo.set_angle(smoothed_angle)

        # ESC update: convert 0-1 throttle to ESC pulse width
        # Assuming ESC neutral = 1560, forward max = 1700
        esc_speed = 1560 + int(throttle_val * 140)  # 1560 + 0~140
        esc.set_speed(esc_speed)
        esc.update()

        print(f"RAW: angle={raw_angle:.2f} throttle={raw_throttle:.2f} | "
              f"ACTUATORS: servo={smoothed_angle:.1f} esc={esc_speed}")

    except Exception as e:
        print("Error processing MQTT message:", e)


# --- MQTT setup ---

client = mqtt.Client()
client.connect("0.0.0.0", 1883, 60)
client.subscribe("donkey/control")
client.on_message = on_message

print("?? Car MQTT controller running...")

try:
    client.loop_forever()
finally:
    servo.cleanup()
    esc.cleanup()
