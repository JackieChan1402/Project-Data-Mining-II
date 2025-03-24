import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset paths
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"


# Function to load YOLO annotations
def load_annotations(anno_path):
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    num_objects = len(lines)
    return num_objects


# Function to extract features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128)).flatten()
    return gray


# Prepare dataset
X, y_objects = [], []
for image_folder in image_folders:
    image_files = sorted(os.listdir(image_folder))
    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(annotation_path):
            continue  # Skip if annotation file doesn't exist

        image = cv2.imread(image_path)
        features = extract_features(image)
        num_objects = load_annotations(annotation_path)

        X.append(features)
        y_objects.append(num_objects)

X = np.array(X)
y_objects = np.array(y_objects)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_objects, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=50,max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train_pca, y_train)

# Save models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(pca, "pca_model.pkl")

# Evaluate model
predictions = rf.predict(X_test_pca)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"MAE: {mae}, MSE: {mse}")
