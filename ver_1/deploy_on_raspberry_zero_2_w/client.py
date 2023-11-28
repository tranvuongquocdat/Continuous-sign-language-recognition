import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import time
import numpy as np
import socket
import pickle
import struct
import cv2
from picamera2 import Picamera2
import threading

class ScreenPrinter:
    def __init__(self):
        # Define the Reset Pin
        self.oled_reset = digitalio.DigitalInOut(board.D4)

        # Display Parameters
        self.WIDTH = 128
        self.HEIGHT = 64
        self.PADDING = 2

        # Use for I2C.
        i2c = board.I2C()
        self.oled = adafruit_ssd1306.SSD1306_I2C(self.WIDTH, self.HEIGHT, i2c, addr=0x3C, reset=self.oled_reset)

        # Create blank image for drawing.
        self.image = Image.new("1", (self.oled.width, self.oled.height))
        self.draw = ImageDraw.Draw(self.image)

        # Load a font
        self.font = ImageFont.truetype('roboto.ttf', 12)

        # Store the last printed text
        self.last_text = None

    def print_screen(self, text):
        # Clear display only if the new text is different from the last text
        if text != self.last_text:
            self.oled.fill(0)
            self.oled.show()
            self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), fill=0)

        max_width = self.WIDTH - 2 * self.PADDING
        bbox = self.draw.textbbox((0,0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        lines = [text]
        if text_width > max_width:
            words = text.split()
            lines = []
            line = ""
            for word in words:
                if self.draw.textbbox((0,0), line + word, font=self.font)[2] <= max_width:
                    line += word + " "
                else:
                    lines.append(line.strip())
                    line = word + " "
            lines.append(line.strip())

        y = (self.HEIGHT - text_height * len(lines)) // 2
        for line in lines:
            bbox = self.draw.textbbox((0,0), line, font=self.font)
            text_width = bbox[2] - bbox[0]
            x = (self.WIDTH - text_width) // 2
            self.draw.text((x, y), line, font=self.font, fill=255)
            y += text_height

        # Display image
        self.oled.image(self.image)
        self.oled.show()

        # Update the last printed text
        self.last_text = text

def send_images(piCam, client_socket):
    while True:
        frame = piCam.capture_array()

        # Resize khung hình
        frame = cv2.resize(frame, (320, 240))

        # Chuyển đổi và nén khung hình thành JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        message_size = struct.pack("L", len(data))
        client_socket.sendall(message_size + data)
        # print("Sent image")  # Added diagnostic print

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def receive_results(client_socket, printer):
    while True:
        # Nhận kết quả từ máy 2
        result_data = client_socket.recv(3).decode()  # Nhận 3 byte, vì biết trước kết quả luôn có 3 ký tự
        if not result_data:
            break

        # print("Received:", result_data)  # Added diagnostic print
        text = "Prediction sign language: " + result_data
        printer.print_screen(text)


if __name__ == "__main__":
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

    # Khởi tạo OLED
    printer = ScreenPrinter()

    # Bắt đầu các luồng
    thread1 = threading.Thread(target=send_images, args=(piCam, client_socket))
    thread2 = threading.Thread(target=receive_results, args=(client_socket, printer))

    thread1.start()
    thread2.start()

    # Join the threads to the main thread
    thread1.join()
    thread2.join()

    piCam.stop()
    cv2.destroyAllWindows()
