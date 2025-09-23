pip install opencv-python==4.10.0.82
import os, cv2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;10000000|max_delay;500000"
cap = cv2.VideoCapture("rtsp://admin:Cmnd024125@192.168.1.10:554/onvif1", cv2.CAP_FFMPEG)
print("opened:", cap.isOpened())
