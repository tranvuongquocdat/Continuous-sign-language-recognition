# AR glass for Sign language recognition

# **Establishing a Server to Execute an AI Model**

Setting up the model:

- Configuration prerequisites:

```jsx
pip install -r requirements.txt
```

- Data Processing and Augmentation:

Execute **`augment_data.py`** for data augmentation. This includes slight skews of 5 degrees to both the left and right, as well as adjustments in lighting conditions among others.

- Feature Extraction from Data:

Derive specific features from the original data. This encompasses elements like human keypoints, optical flow, and more.

- Model Construction and Training:

Construct a LeNet5-based AI model tailored for classifying the extracted data. Subsequently, initiate training for a duration of 50 epochs. This procedure is estimated to consume up to 2 hours, culminating in the generation of a .pt file that encapsulates both the model and its architecture.

- Model Evaluation:
    - Execute the testing script and document instances where predictions were not accurate.

Setting up the Server for Client Connections:

- Server IP Configuration:
Given the need for local connectivity, the server will utilize the IP address “0.0.0.0” and will listen on port “123”. This will be established using a TCP-IP protocol via sockets.
- Launch and Operate Server:
The server will be coded and set in a listening mode, awaiting connection requests from potential clients.

# Build client with raspberry pi zero 2:

**Raspberry Pi Zero 2 Configuration:**

1. **Download and Install Raspberry Pi OS:**
    - **Download Raspberry Pi OS**:
    Head over to [Raspberry Pi OS – Raspberry Pi](https://www.raspberrypi.com/software/) to download the latest Raspberry Pi OS image.
    - **Write the OS Image to the microSD card**:
    You can use software like Balena Etcher or Raspberry Pi Imager. After installing the software:
        - Insert your microSD card into the card reader of your computer.
        - Open the imaging software and select the downloaded Raspberry Pi OS image.
        - Select the appropriate drive for your microSD and then click on 'Write' or 'Flash' to start the writing process.
2. **Initial Setup of Raspberry Pi:**
    - **Insert microSD Card**: Once the OS is written to the microSD card, insert it into the Raspberry Pi Zero 2's microSD slot.
    - **Connect Peripherals**: Connect necessary peripherals like keyboard, mouse, monitor, and power supply. If you are using a mini HDMI port or micro USB for peripherals, make sure you have the required adapters.
    - **Power On**: Plug in the power supply to turn on the Raspberry Pi. You should see a series of boot messages, and then the Raspberry Pi OS desktop will appear.
3. **Basic Configuration with raspi-config:**
    - **Open Terminal**: Click on the terminal icon or use the shortcut `CTRL` + `ALT` + `T` to open the terminal.
    - **Access Configuration Tool**: Type in the following command to access the Raspberry Pi configuration tool:
        
        ```bash
        sudo raspi-config
        ```
        
    - **Navigate through Options**: Here you can:
        - Expand the filesystem to use the entire microSD card.
        - Change the default user password.
        - Configure the boot options, like booting to desktop or CLI.
        - Set up localization options such as timezone, keyboard layout, and language.
        - Enable or disable interfaces like SSH, SPI, I2C, etc.
        - And more...
    - **Finish and Reboot**: Once done with the configurations, select 'Finish' and reboot the Raspberry Pi for the changes to take effect.
4. **Network Configuration**:
    - **WiFi Setup**: If you're using WiFi, click on the network icon on the top right of the desktop, select your network, and enter the WiFi password.
    - **Ethernet**: If you're connecting via Ethernet, the Raspberry Pi should automatically detect the network and establish a connection.
5. **Update and Upgrade**:
    
    It's a good practice to ensure your Raspberry Pi is updated with the latest packages. In the terminal, enter:
    
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
    

**Camera Configuration for Raspberry Pi Zero 2:**

1. **Hardware Setup**:
    - **Unpack the Camera**: Carefully unpack the Raspberry Pi Camera Module from its box.
    - **Link the camera:**
    
    [Raspberry Pi Camera Module 3 - 12MP - Ống kính Lấy nét Tự động](https://www.cytrontech.vn/p-raspberry-pi-camera-module-3-12mp-with-auto-focus?currency=VND&gad=1&gclid=CjwKCAjw-eKpBhAbEiwAqFL0mhVkmg8NaS4tAsMjBAkK-kvouyhNpbKvn2hF3Lg3q-7_YtPTlrNeXxoCPegQAvD_BwE)
    
    - **Camera Connection**:
        - Ensure the Raspberry Pi is turned off.
        - Lift the camera connector on the Raspberry Pi (the plastic tab should be pulled upwards).
        - Slide the camera ribbon cable with the silver contacts facing the HDMI port and ensure it's firmly seated.
        - Press the plastic tab back down.
    - **Position the Camera**: Place the camera in a suitable position or mount depending on your use-case.

![Untitled](Images/Untitled.png)

1. **Software Configuration**:
    - **Enable the Camera Interface**:
        - Access the Raspberry Pi configuration tool by typing `sudo raspi-config` in the terminal.
        - Navigate to `Interface Options`, then select `Camera` and enable it.
        - Exit and reboot your Raspberry Pi for the changes to take effect.
2. **Test the Camera**:
    - After the Raspberry Pi has rebooted, open the terminal and type the following command to capture an image:
        
        ```bash
        raspistill -o test.jpg
        
        ```
        
    - If successful, this command will capture an image and save it as `test.jpg` in the current directory. You can view it using the default image viewer or transfer it to another machine for viewing.
3. **Capture Video**:
    - You can also capture video using the `raspivid` command. To capture a 10-second video, use:
        
        ```bash
        raspivid -o test_video.h264 -t 10000
        
        ```
        
    - This command will save a 10-second video as `test_video.h264`. You might need additional software or conversion to view this video on certain platforms.
4. **Advanced Configuration**:
    
    The Raspberry Pi camera module offers a variety of settings like exposure, ISO, effects, and more. You can explore more options and settings using `raspistill` and `raspivid` by typing `raspistill --help` or `raspivid --help` in the terminal.
    
5. **Python Integration**:
    
    If you're developing an application that requires Python integration with the camera, consider using the `picamera` Python library. Install it with:
    
    ```bash
    sudo apt install python3-picamera
    
    ```
    
    This library provides a pure Python interface for the camera module, allowing for more advanced use cases, like real-time image processing and integration with other Python tools.
    

By following the steps above, you should have a working camera setup with your Raspberry Pi Zero 2. Depending on the specific requirements of your project, you can further customize and automate the camera functionalities.

**OLED 0.96 inch config:**

The 0.96 inch OLED display typically interfaces with the Raspberry Pi using the I2C communication protocol. Here's a step-by-step guide on how to connect and configure the OLED display for the Raspberry Pi Zero 2:

1. **Hardware Connection:**
    
    Connect the OLED pins to the Raspberry Pi Zero 2 as follows:
    
    - VCC to 3.3V
    - GND to GND
    - SCL to GPIO3 (SCL)
    - SDA to GPIO2 (SDA)
        
        ![Untitled](Images/Untitled 1.png)
        
2. **Enable I2C:**
    
    Before using the OLED, you need to enable the I2C interface on the Raspberry Pi.
    
    - Open a terminal and type the following command:
        
        ```jsx
        sudo raspi-config
        
        ```
        
    - Navigate to `Interface Options`, then select `I2C` and enable it.
    - Exit and reboot your Raspberry Pi for the changes to take effect.
3. **Install Necessary Libraries:**
    
    To interface with the OLED display, you'll need some specific Python libraries.
    
    - Install the required packages:
        
        ```jsx
        sudo apt-get update
        sudo apt-get install python3-pip python3-pil python3-numpy
        sudo pip3 install Adafruit-SSD1306
        
        ```
        
4. **Write a Test Script:**
    
    Create a simple script to test if the OLED is working. Let's call it `oled_test.py`.
    
    ```python
    import Adafruit_SSD1306
    from PIL import Image, ImageDraw, ImageFont
    
    disp = Adafruit_SSD1306.SSD1306_128_64(rst=None)
    disp.begin()
    disp.clear()
    disp.display()
    
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))
    
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    
    font = ImageFont.load_default()
    draw.text((10, 30), "Hello, Pi Zero 2!", font=font, fill=255)
    
    disp.image(image)
    disp.display()
    
    ```
    
    - Run the script:
        
        ```jsx
        python3 oled_test.py
        
        ```
        
    
    If everything was set up correctly, you should see "Hello, Pi Zero 2!" displayed on your OLED screen.
    

Config socket:

In this part, first we need to find out the server IP and port: here the server IP is 192.168.1.10, locally connection, port “123”

**Client Setup on Raspberry Pi**:

Assuming you're running a Python environment on your Raspberry Pi, follow these steps:

- **Python Script**:
    
    Create a file named **`client.py`** on the Raspberry Pi and input the following script:
    
    ```python
    pythonCopy code
    import socket
    
    SERVER_IP = "192.168.1.10"
    PORT = 123
    
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    client_socket.connect((SERVER_IP, PORT))
    print(f"[*] Connected to server at {SERVER_IP}:{PORT}")
    
    message = client_socket.recv(1024)
    print(f"Received message: {message.decode('utf-8')}")
    
    client_socket.close()
    
    ```
    
- **Run the Client**:
    
    On the Raspberry Pi's terminal (or SSH session if you're accessing remotely), navigate to the directory where **`client.py`** is located and run:
    
    ```bash
    bashCopy code
    python3 client.py
    
    ```
    
    The client script will now attempt to connect to the specified server IP and port. Upon successful connection, it will receive a message from the server and display it.
    

# Design AR glass’s case