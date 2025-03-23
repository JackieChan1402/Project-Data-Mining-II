import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📌 Định nghĩa đường dẫn dataset
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"  # Thư mục YOLO annotations

# 📌 Cấu hình xử lý ảnh
image_size = (64, 64)  # Resize ảnh về 64x64 pixels
max_bbox = 94  # Số lượng bbox tối đa
num_bbox_features = max_bbox * 4  # Mỗi bbox có 4 feature (x_center, y_center, width, height)

X, y = [], []

# 📌 Đọc ảnh & YOLO annotations
for folder in image_folders:
    for image_name in os.listdir(folder):
        if image_name.endswith((".jpg", ".png")):
            # Đọc ảnh
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️ Lỗi đọc ảnh: {image_path}")
                continue

            image = cv2.resize(image, image_size)
            img_vector = image.flatten()  # Chuyển ảnh thành vector 1D (4096 features)

            # Đọc annotation YOLO
            annotation_path = os.path.join(annotation_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            bbox_data = []

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:max_bbox]:  # Chỉ lấy tối đa max_bbox bbox
                        values = line.strip().split()
                        bbox = list(map(float, values[1:]))  # Lấy x_center, y_center, width, height
                        bbox_data.extend(bbox)

            # Padding nếu bbox < max_bbox
            if len(bbox_data) < num_bbox_features:
                bbox_data.extend([0] * (num_bbox_features - len(bbox_data)))

            # Kết hợp ảnh + bbox features
            feature_vector = np.hstack((img_vector, bbox_data))
            X.append(feature_vector)

            # Gán nhãn từ class đầu tiên trong YOLO annotation
            y.append(int(lines[0].split()[0]) if len(lines) > 0 else 0)

# 📌 Chuyển thành numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 📌 Kiểm tra kích thước dữ liệu
print(f"📌 Tổng số ảnh: {len(X)}")
print(f"📌 Kích thước dữ liệu X: {X.shape}")
print(f"📌 Kích thước dữ liệu y: {y.shape}")

# 📌 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Giảm chiều bằng PCA
pca = PCA(n_components=150)  # Giữ 150 thành phần chính
X_pca = pca.fit_transform(X_scaled)

# 📌 Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 📌 Huấn luyện Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

# 📌 Dự đoán với Decision Tree
y_pred_dt = dt_model.predict(X_test)

# 📌 Đánh giá hiệu suất
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\n📌 Decision Tree Results:")
print("Accuracy:", accuracy_dt)
print(classification_report(y_test, y_pred_dt))

# 📌 Vẽ Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree")
plt.savefig("Confusion Matrix Decision Tree.png")

# 📌 Vẽ biểu đồ phân bố dữ liệu theo lớp
plt.figure(figsize=(8, 5))
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts, color='blue', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Distribution of number of images in each class")
plt.xticks(unique)
plt.savefig("Distribution of number of images in each class DT.png")

# 📌 Vẽ biểu đồ độ chính xác theo từng lớp
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(8, 5))
plt.bar(unique, class_accuracies, color='green', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Accuracy per Class")
plt.title("Grade-wise accuracy - Decision Tree")
plt.xticks(unique)
plt.ylim(0, 1)
plt.savefig("Accuracy per class DT.png")
