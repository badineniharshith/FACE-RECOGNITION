from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Text-to-speech function
def speak(str1):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

# Initialize webcam
video = cv2.VideoCapture(0)

# Load face detector
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load training data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("background.JPEG")
if imgBackground is None:
    print("Warning: 'background.jpg' not found. Displaying plain video frame.")
    use_background = False
else:
    use_background = True

# Ensure Attendance folder exists
os.makedirs("Attendance", exist_ok=True)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    attendance = None  # Reset attendance for each frame

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [str(output[0]), str(timestamp)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    # Show frame in background or as-is
    if use_background:
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)
    else:
        cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    # 'p' to mark attendance
    if k == ord('p') and attendance:
        speak("Attendance Taken.")
        filename = f"Attendance/Attendance_{date}.csv"
        file_exists = os.path.isfile(filename)

        mode = 'a' if file_exists else 'w'
        with open(filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

        time.sleep(2)  # Prevent accidental multiple captures

    # 'q' to quit
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
