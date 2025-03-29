import serial
import time
import json
import random

# Configure serial port
try:
    ser = serial.Serial('/dev/tty.usbserial-0001', 115200, timeout=1)
except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
    exit(1)

# Initial motor data (will be updated in loop)
motor_data = {
    'rightFront': 1,
    'leftFront': 1,
    'rightBack': 1,
    'leftBack': 1,
}

print("Starting direct serial writer...")
try:
    while True:
        # Randomly assign 0 or 1 to each motor
        motor_data['rightFront'] = random.randint(0, 1)
        motor_data['leftFront'] = random.randint(0, 1)
        motor_data['rightBack'] = random.randint(0, 1)
        motor_data['leftBack'] = random.randint(0, 1)

        # Convert motor data to JSON string
        data = json.dumps(motor_data)
        # Write data to ESP32 with a newline
        ser.write(f"{data}\n".encode())
        print(f"Sent to ESP32: {data}")
        
        # Wait 2 seconds before sending again
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    ser.close()  # Ensure the port is closed on exit
    print("Serial port closed")