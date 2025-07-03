import cv2
import datetime
import os

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')  # Ensure this file is in your script's directory

# Create directory to save photos
output_dir = "smile_captures"
os.makedirs(output_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect smiles within the face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            
            if not captured:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"smile_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Smile detected! Photo saved as {filename}")
                captured = True
            break

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit on pressing 'q'
        break
    elif key == ord('r'):  # Reset capture flag to allow another smile capture
        captured = False

cap.release()
cv2.destroyAllWindows()
