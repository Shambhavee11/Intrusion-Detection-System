import cv2
import torch
import numpy as np
import pygame
import os
from twilio.rest import Client
import time

# Twilio setup
twilio_account_sid = '----'  # Replace with your Twilio Account SID
twilio_auth_token = '------'      # Replace with your Twilio Auth Token
twilio_phone_number = '--------' # Replace with your Twilio phone number
target_phone_number = '-------'  # Replace with the recipient's phone number

client = Client(twilio_account_sid, twilio_auth_token)

# Don't forget to double check the paths
# Path to the alarm sound
path_alarm = r"C:\Users\shamb\OneDrive\Desktop\target-detector-yolov5-main\target-detector-yolov5-main\Alarm\alarm.wav"

# Initializing pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use the absolute path for the video file
video_path = r"C:\Users\shamb\OneDrive\Desktop\target-detector-yolov5-main\target-detector-yolov5-main\Test Videos\thief_video2.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

target_classes = ['car', 'bus', 'truck', 'person']
count = 0
number_of_photos = 3
pts = []  # Polygon points

# Function to draw polygon (ROI)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result == 1

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

# Function to send SMS using Twilio
def send_sms(message):
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=target_phone_number
    )

# Function to make a call using Twilio
def make_call():
    call = client.calls.create(
        url='http://demo.twilio.com/docs/voice.xml',  # URL for Twilio to fetch call instructions
        to=target_phone_number,
        from_=twilio_phone_number
    )

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_detected = frame.copy()
    frame = preprocess(frame)
    results = model(frame)

    # Using pandas to get the detected objects' data
    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # Write name
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # Draw center
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Drawing the polygon
        if len(pts) >= 4:
            frame_copy = frame.copy()
            cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
            if center_x is not None and center_y is not None:
                # Checking if the center of the object is inside the polygon and if the object is a person
                if inside_polygon((center_x, center_y), np.array([pts])) and name == 'person':
                    mask = np.zeros_like(frame_detected)
                    points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    points = points.reshape((-1, 1, 2))
                    mask = cv2.fillPoly(mask, [points], (255, 255, 255))             
                    frame_detected = cv2.bitwise_and(frame_detected, mask)

                    # Saving the detected image
                    if count < number_of_photos:
                        cv2.imwrite("Detected Photos/detected" + str(count) + ".jpg", frame_detected)
                        count += 1

                    # Playing the alarm
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                    
                    # Send SMS
                    message = "Intruder detected: A person was detected in the restricted area."
                    send_sms(message)
                    print("SMS sent.")

                    # Wait for 10 seconds before making a call
                    time.sleep(10)

                    # Make a call
                    make_call()
                    print("Call initiated.")

                    cv2.putText(frame, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
