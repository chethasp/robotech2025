import serial
import time
import json

# Configure serial port
try:
    ser = serial.Serial('/dev/tty.usbserial-0001', 115200, timeout=1)
except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
    exit(1)

# Simulated motor data (you can modify this as needed)
motor_data = {
    'speed': 100,
}

print("Starting direct serial writer...")
try:
    while True:
        # Convert motor data to JSON string
        data = json.dumps(motor_data)
        # Write data to ESP32 with a newline
        ser.write(f"{data}\n".encode())
        print(f"Sent to ESP32: {data}")
        
        # Wait 5 seconds before sending again
        time.sleep(5)
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    ser.close()  # Ensure the port is closed on exit
    print("Serial port closed")
    
# robot MAC addr
# 94:54:C5:74:DC:08