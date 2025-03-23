import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 📌 Thư mục chứa dữ liệu
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"

# 📌 Cấu hình HOG
hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)}

# 📌 Cấu hình bounding box
max_bbox = 40
num_bbox_features = max_bbox * 4

X, image_paths = [], []  # Lưu đường dẫn ảnh để hiển thị

# 📌 Duyệt qua ảnh trong các thư mục
for folder in image_folders:
    for image_name in os.listdir(folder):
        if image_name.endswith((".jpg", ".png")):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"⚠️ Lỗi đọc ảnh: {image_path}")
                continue

            image_resized = cv2.resize(image, (64, 64))

            # 📌 Trích xuất đặc trưng HOG
            hog_features = hog(image_resized, **hog_params)
            hog_features = hog_features.flatten()

            # 📌 Đọc annotation YOLO
            annotation_path = os.path.join(annotation_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            bbox_data = []

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:max_bbox]:  # Chỉ lấy tối đa max_bbox bbox
                        values = line.strip().split()
                        bbox = list(map(float, values[1:]))  # Lấy x_center, y_center, width, height
                        bbox_data.extend(bbox)

            # 📌 Padding nếu bbox < max_bbox
            if len(bbox_data) < num_bbox_features:
                bbox_data.extend([0.0] * (num_bbox_features - len(bbox_data)))

            bbox_data = np.array(bbox_data, dtype=np.float32)[:num_bbox_features]

            # 📌 Kết hợp feature ảnh + bbox
            feature_vector = np.hstack((hog_features, bbox_data))

            X.append(feature_vector)
            image_paths.append(image_path)

# 📌 Chuyển thành numpy array
X = np.array(X, dtype=np.float32)

# 📌 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Giảm chiều bằng PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# 📌 Áp dụng KMeans Clustering
num_clusters = 5  # Số cụm
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 📌 Hiển thị số lượng ảnh trong mỗi cụm
unique, counts = np.unique(clusters, return_counts=True)
cluster_info = dict(zip(unique, counts))
print("📌 Số lượng ảnh trong mỗi cụm:", cluster_info)

# 📌 Vẽ biểu đồ phân bố cụm
plt.figure(figsize=(8, 5))
plt.bar(cluster_info.keys(), cluster_info.values(), color='blue', alpha=0.7)
plt.xlabel("Cluster ID")
plt.ylabel("Số lượng ảnh")
plt.title("Phân phối số lượng ảnh trong các cụm")
plt.xticks(range(num_clusters))
plt.show()

# 📌 Hiển thị một số ảnh từ mỗi cụm
fig, axes = plt.subplots(num_clusters, 5, figsize=(12, num_clusters * 3))

for cluster_id in range(num_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0][:5]  # Chọn 5 ảnh từ cụm này
    for i, idx in enumerate(cluster_indices):
        image = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue  # Bỏ qua ảnh nếu bị lỗi

        axes[cluster_id, i].imshow(image, cmap='gray')
        axes[cluster_id, i].axis("off")
        axes[cluster_id, i].set_title(f"Cluster {cluster_id}")

plt.tight_layout()
plt.show()
