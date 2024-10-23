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

GPS:
step1:- update RaspberryPI
pi@raspberrypi:-$ sudo apt-get update/upgrade
step2:- edit the boot/config.text file
pi@raspberrypi:-$ sudo nano/boot/config.txt
dtparam=spi=on 
dtoverlay=pi3-disable-bt 
core freq=250 
enable_uart=1 
force_turbo-1
step3 :-Reboot RaspberryPi using the command sudo reboot
step 4:-stop and disable the pi's serial ttySO service
sudo systemctl stop serial-getty@ttyS0.service 
sudo systemctl disable serial-getty@ttyS0.service
The following commands can be used to enable it again if needed
sudo systemctl start serial-getty@ttyS0.service 
sudo systemctl enable serial-getty@ttyS0.service
step5 :-Reboot RaspberryPi using the command sudo reboot
Step6:Enable the ttyAMA0 services
pi@raspberrypi:- sudo systemctl enable serial-getty@ttyAMA0.service
Step 7:Install minicom 
pi@raspberrypi:-sudo apt-get install minicom
now run the following command:
pi@raspberrypi:- sudo cat/dev/ttyAMA0
