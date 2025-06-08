import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import glob  # For file path matching
import cv2  # OpenCV for image processing
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
import numpy as np  # Numpy for array manipulation
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent Classifier
from sklearn.metrics import accuracy_score  # To calculate accuracy of the model
from joblib import dump  # To save the trained model



# Define the parameter grid for Grid Search
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [1000, 2000, 3000],
    'eta0': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling']
}

# Initialize lists for storing image data and corresponding labels
data = []
labels = []

# Function to detect faces in an image using MTCNN
def face_detector(img):
    try:
        detector = MTCNN()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)
        if faces:
            out = faces[0]
            x, y, w, h = out["box"]
            return img[y:y+h, x:x+w]  # Return the cropped face region
    except Exception as e:
        print(f"Error detecting face: {e}")
    return None  # If an error occurs or no face is detected

# Check if preprocessed data already exists
if os.path.exists("preprocessed_data.npy") and os.path.exists("preprocessed_labels.npy"):
    print("[info] Loading preprocessed data...")
    data = np.load("preprocessed_data.npy")
    labels = np.load("preprocessed_labels.npy")
else:
    print("[info] Preprocessing data...")

    # Path to the dataset
    folder_path = Path(r"E:\machinlerning\Laugh_detection\Q3\smile_dataset")
    image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

    # Process each image
    for i, item in enumerate(image_paths):
        img = cv2.imread(item)
        face = face_detector(img)
        if face is None:
            continue
        face = cv2.resize(face, (32, 32))
        face = face.flatten() / 255.0  # Flatten and normalize pixel values
        data.append(face)
        label = item.split("\\")[-2]  # Extract label from the file path
        labels.append(label)
        if i % 100 == 0:
            print(f"[info] : {i}/{len(image_paths)} processed")

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    print("[info] Saving preprocessed data...")
    np.save("preprocessed_data.npy", data)
    np.save("preprocessed_labels.npy", labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize the SGDClassifier and Grid Search
clf = SGDClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model using Grid Search
grid_search.fit(x_train, y_train)

# Output the best parameters and score found by Grid Search
print("Best Parameters found: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# Get the best estimator from the grid search
best_clf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_clf.predict(x_test)

# Calculate and print the final accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy: ", accuracy)

# Save the classifier model for future use
dump(best_clf, "Laugh_detection.z")

