import numpy as np
import socket
import struct
import cv2
from picamera2 import Picamera2

# Khởi tạo PiCamera
piCam = Picamera2()
piCam.preview_configuration.main.size = (640, 480)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

# Khởi tạo socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.10', 123))
print("Đã kết nối thành công đến server!")

# Kích thước mong muốn sau khi resize
#desired_size = (640, 480)

while True:
    frame = piCam.capture_array()

    # Resize khung hình
    frame = cv2.resize(frame, (640, 480))

    # Chuyển đổi và nén khung hình thành JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    message_size = struct.pack("L", len(data))
    client_socket.sendall(message_size + data)

    # Nhận kết quả từ máy 2
    result = client_socket.recv(1024).decode()
    print("Kết quả từ máy 2:", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

piCam.stop()
cv2.destroyAllWindows()
