git clone https://github.com/sasukee004/led
python blink.py

git clone https://github.com/sasukee004/4seg
python clock.py

import time
import picamera
camera = picamera.PiCamera()
camera.resolution(1024, 768)
camera.start_preview()
time.sleep(2)
camera.capture('test.jpg')
camera.stop_preview()

sudo apt-get install python-picamera
sudo apt-get install python3-picamera
