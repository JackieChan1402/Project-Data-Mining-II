import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📌 Danh sách thư mục chứa ảnh
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"  # Thư mục YOLO annotations

# 📌 Cấu hình xử lý ảnh
image_size = (64, 64)  # Resize ảnh về kích thước cố định
max_bbox = 94  # Mở rộng tối đa 94 bounding boxes
num_bbox_features = max_bbox * 4  # Mỗi bbox có 4 đặc trưng (x_center, y_center, width, height)

X, y = [], []

# 📌 Đọc ảnh & annotation YOLO từ nhiều thư mục
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
            img_vector = image.flatten()  # Chuyển ảnh thành vector (4096 features)

            # Tìm file annotation tương ứng
            annotation_path = os.path.join(annotation_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            bbox_data = []

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:max_bbox]:  # Lấy tối đa `max_bbox` boxes
                        values = line.strip().split()
                        bbox = list(map(float, values[1:]))  # Lấy x_center, y_center, width, height
                        bbox_data.extend(bbox)

            # 📌 Đảm bảo số lượng feature bbox cố định (94 bbox → 376 feature)
            if len(bbox_data) < num_bbox_features:
                bbox_data.extend([0] * (num_bbox_features - len(bbox_data)))  # Padding nếu thiếu

            # 📌 Kết hợp đặc trưng ảnh + bbox
            feature_vector = np.hstack((img_vector, bbox_data))
            X.append(feature_vector)

            # 📌 Lấy nhãn từ class đầu tiên trong file YOLO (nếu có nhiều, chỉ lấy class đầu)
            y.append(int(lines[0].split()[0]) if len(lines) > 0 else 0)

# 📌 Chuyển đổi dữ liệu thành numpy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 📌 Kiểm tra kích thước dữ liệu sau khi hợp nhất
print(f"📌 Tổng số ảnh từ 3 thư mục: {len(X)}")
print(f"📌 Kích thước dữ liệu X: {X.shape}")  # (Số ảnh, Số đặc trưng)
print(f"📌 Kích thước dữ liệu y: {y.shape}")  # (Số ảnh,)

# 📌 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Giảm chiều bằng PCA
pca = PCA(n_components=150)  # Giữ lại 150 thành phần chính
X_pca = pca.fit_transform(X_scaled)

# 📌 Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 📌 Huấn luyện SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# 📌 Dự đoán
y_pred = svm_model.predict(X_test)

# 📌 Đánh giá hiệu suất
accuracy = accuracy_score(y_test, y_pred)
print("📌 Độ chính xác:", accuracy)
print("\n📌 Báo cáo phân loại:\n", classification_report(y_test, y_pred))

# 📌 Vẽ Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("SVM Classification Report.png")

# 📌 Vẽ biểu đồ so sánh số lượng mẫu trong từng lớp
plt.figure(figsize=(8, 5))
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts, color='blue', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Distribution of number of images in each class")
plt.xticks(unique)
plt.savefig("SVM Distribution of number of images in each class.png")

# 📌 Vẽ biểu đồ Accuracy của các lớp
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(8, 5))
plt.bar(unique, class_accuracies, color='green', alpha=0.7)
plt.xlabel("Class")
plt.ylabel("Accuracy per Class")
plt.title("Grade-wise accuracy")
plt.xticks(unique)
plt.ylim(0, 1)
plt.savefig("SVM Grade-wise Accuracy.png")
