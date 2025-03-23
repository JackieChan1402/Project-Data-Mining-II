import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📌 Define dataset paths
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"  # Folder containing YOLO annotations

# 📌 Image processing settings
image_size = (64, 64)  # Resize all images to 64x64
max_bbox = 94  # Maximum number of bounding boxes per image
num_bbox_features = max_bbox * 4  # 94 boxes * 4 features (x_center, y_center, width, height)

X, y = [], []

# 📌 Read images & YOLO annotations
for folder in image_folders:
    for image_name in os.listdir(folder):
        if image_name.endswith((".jpg", ".png")):
            # Read image
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️ Error reading image: {image_path}")
                continue

            image = cv2.resize(image, image_size)
            img_vector = image.flatten()  # Convert to 1D feature vector (4096 features)

            # Read annotation (YOLO format)
            annotation_path = os.path.join(annotation_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            bbox_data = []

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:max_bbox]:  # Read up to 94 bounding boxes
                        values = line.strip().split()
                        bbox = list(map(float, values[1:]))  # Extract x_center, y_center, width, height
                        bbox_data.extend(bbox)

            # Ensure a fixed number of bounding box features (94 boxes → 376 features)
            if len(bbox_data) < num_bbox_features:
                bbox_data.extend([0] * (num_bbox_features - len(bbox_data)))  # Padding if fewer than 94 boxes

            # Combine image + bbox features
            feature_vector = np.hstack((img_vector, bbox_data))
            X.append(feature_vector)

            # Assign class label from the first bounding box
            y.append(int(lines[0].split()[0]) if len(lines) > 0 else 0)

# 📌 Convert data to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 📌 Check dataset size
print(f"📌 Total images: {len(X)}")
print(f"📌 Feature matrix size: {X.shape}")  # (num_samples, num_features)
print(f"📌 Label vector size: {y.shape}")  # (num_samples,)

# 📌 Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Reduce dimensions using PCA
pca = PCA(n_components=150)  # Reduce to 150 components
X_pca = pca.fit_transform(X_scaled)

# 📌 Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 📌 Train KNN classifier
k_neighbors = 5  # Number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(X_train, y_train)

# 📌 Make predictions & evaluate
y_pred = knn_model.predict(X_test)

# 📌 Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("📌 Accuracy:", accuracy)
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

# 📌 Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN")
plt.savefig("Confusion Matrix KNN .png")

# 📌 Plot class distribution
plt.figure(figsize=(8, 5))
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts, color='blue', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Dataset")
plt.xticks(unique)
plt.savefig("Class Distribution KNN .png")

# 📌 Plot per-class accuracy
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(8, 5))
plt.bar(unique, class_accuracies, color='green', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Accuracy per Class")
plt.title("Per-Class Accuracy - KNN")
plt.xticks(unique)
plt.ylim(0, 1)
plt.savefig("Per-Class Accuracy KNN .png")
