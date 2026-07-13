
from pathlib import Path
import cv2
import numpy as np
import sys

# Handle PyInstaller environment
if hasattr(sys, '_MEIPASS'):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

dataset_path = BASE_DIR / "data"

dataset_path.mkdir(exist_ok=True)

# Get name argument
if len(sys.argv) > 1:
    person_name = sys.argv[1].strip()
    print(f"Capturing faces for {person_name}...")
else:
    print(" No name provided. Please run from the GUI or provide a name argument.")
    sys.exit()

# Load Haar cascade face detector (ships with OpenCV -- no external files needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Failed to load the face detector.")
    sys.exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam. Try changing the camera index.")
    sys.exit()

face_data = []
count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Process every 5th frame for speed
    if frame_count % 5 != 0:
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Detect faces with the Haar cascade (needs a grayscale image)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, fw, fh) in faces:
        face_section = frame[y:y + fh, x:x + fw]
        if face_section.size == 0:
            continue

        face_section = cv2.cvtColor(face_section, cv2.COLOR_BGR2GRAY)
        face_section = cv2.equalizeHist(face_section)
        face_section = cv2.resize(face_section, (128, 128))

        face_data.append(face_section)
        count += 1

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or count >= 200:
        break

# Save face data
face_data = np.array(face_data)
np.save(dataset_path / f"{person_name}.npy", face_data)
print(f"Saved {face_data.shape} for {person_name} in {dataset_path}")

cap.release()
cv2.destroyAllWindows()
