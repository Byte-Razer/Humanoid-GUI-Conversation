

from pathlib import Path
import numpy as np
import cv2
import pyttsx3
import sys
import time


BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "data"


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())
    

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]



# Load Haar cascade face detector (ships with OpenCV -- no external files needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Failed to load the face detector.")
    sys.exit()


face_data = []
labels = []
names = {}
class_id = 0

if not dataset_path.exists():
    print("'data' folder not found. Please run train.py first.")
    sys.exit()

for file in dataset_path.glob("*.npy"):
    names[class_id] = file.stem
    print(" Loaded:", file.name)

    data_item = np.load(file)
    data_item = data_item.reshape(data_item.shape[0], -1)
    face_data.append(data_item)

    target = class_id * np.ones((data_item.shape[0],))
    labels.append(target)
    class_id += 1

if not face_data:
    print("No training data found in ./data/. Please collect faces first.")
    sys.exit()

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("\n Training data loaded successfully!")
print("   Face dataset shape:", face_dataset.shape)
print("   Face labels shape:", face_labels.shape)


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)


USE_LBPH = True   # Set to False to disable LBPH

# LBPH returns a distance score for the best match -- LOWER means a better match.
# A face whose best score is above this cutoff is treated as "Unknown".
# Raise it if a known person shows as Unknown; lower it if strangers get matched.
RECOGNITION_THRESHOLD = 110

if USE_LBPH:
    print("\nInitializing LBPH recognizer (tuned parameters)...")
    # No built-in threshold here -- we apply RECOGNITION_THRESHOLD ourselves so
    # predict() always returns the nearest label together with its distance.
    lbph = cv2.face.LBPHFaceRecognizer_create(
        radius=1,        # Slightly larger radius for lighting robustness
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )
    # Prepare data for LBPH training (grayscale images)
    faces, face_ids = [], []
    for file in dataset_path.glob("*.npy"):
        current_id = [key for key, name in names.items() if name == file.stem][0]
        data = np.load(file)
        for img in data:
            if len(img.shape) == 3:  # Color image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = img

            gray = cv2.equalizeHist(gray)
            faces.append(gray)
            face_ids.append(current_id)
    if faces:
        lbph.train(faces, np.array(face_ids))
        print("LBPH training complete!")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam. Try changing the camera index.")
    sys.exit()

print("\nPress 'q' to quit.\n")

recognized = False       # set once we greet a known person
recognized_time = None   # when we greeted, so we can keep the live view for 10s
LINGER_SECONDS = 10

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces with the Haar cascade (needs a grayscale image)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, fw, fh) in faces:
        x1, y1, x2, y2 = x, y, x + fw, y + fh

        face_section = frame[y1:y2, x1:x2]
        if face_section.size == 0:
            continue

        # -----------------------------
        # Normalize face
        # -----------------------------
        face_section = cv2.cvtColor(face_section, cv2.COLOR_BGR2GRAY)
        face_section = cv2.equalizeHist(face_section)
        face_section = cv2.resize(face_section, (128, 128))

        # -----------------------------
        # Predict using LBPH or KNN
        # -----------------------------
        if USE_LBPH:
            label, confidence_value = lbph.predict(face_section)
            if label >= 0 and confidence_value <= RECOGNITION_THRESHOLD:
                pred_name = names.get(label, "Unknown")
            else:
                pred_name = "Unknown"
            label_text = f"{pred_name} ({confidence_value:.0f})"
        else:
            out = knn(trainset, face_section.flatten())
            pred_name = names[int(out)]
            label_text = pred_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label_text,
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

        if not recognized and pred_name != "Unknown":
            engine.say(f"Hi {pred_name}. Welcome to Utpal Shangvi Global School!")
            engine.runAndWait()
            recognized = True
            recognized_time = time.time()
            break  # stop scanning the remaining faces in this frame

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # After greeting, keep the live (smooth) video running for a few more
    # seconds, then turn the camera off.
    if recognized and time.time() - recognized_time >= LINGER_SECONDS:
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

