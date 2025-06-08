import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import load

def face_detector(img):
    try:
        detector = MTCNN()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)
        if faces:
            x, y, w, h = faces[0]["box"]
            return img[y:y+h, x:x+w], x, y, w, h
        else:
            return None, None, None, None, None
    except Exception as e:
        print(f"Error detecting face: {e}")
        return None, None, None, None, None

# Load pre-trained classifier
clf = load(r"E:\machinlerning\Laugh_detection\Laugh_detection.z")

# Path to the folder containing images
folder_path = Path(r"E:\machinlerning\Laugh_detection\Q3\test_data")

# Find all images with jpg and png extensions
image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

# Process each image
for i, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    face, x, y, w, h = face_detector(img)

    if face is None:
        continue

    # Preprocess the face
    face = cv2.resize(face, (32, 32)).flatten() / 255.0

    # Predict smile or not
    out = clf.predict(np.array([face]))[0]

    # Draw rectangle and put text based on the prediction
    if out == "neg":
        color = (0, 0, 255)  # Red for no smile
        label = "The face doesn't smile"
    elif out == "pos":
        color = (0, 255, 0)  # Green for smile
        label = "The face smiles"

    # Draw rectangle and label on the image
    if x is not None and y is not None:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

    # Show the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

# Destroy all windows after processing
cv2.destroyAllWindows()
