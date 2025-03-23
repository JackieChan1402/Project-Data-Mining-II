import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler

# 📌 Thư mục chứa dữ liệu
image_folders = ["PART_1/images", "PART_2/PART_2/images", "PART_3/PART_3/images"]
annotation_folder = "PART_1/6categories"

# 📌 Cấu hình HOG
hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)}

# 📌 Cấu hình bounding box
max_bbox = 40
num_bbox_features = max_bbox * 4

X, y = [], []
bbox_counts = []  # Lưu số lượng bbox trên mỗi ảnh

# 📌 Duyệt qua ảnh
for folder in image_folders:
    for image_name in os.listdir(folder):
        if image_name.endswith((".jpg", ".png")):
            # 📌 Đọc ảnh
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"⚠️ Lỗi đọc ảnh: {image_path}")
                continue

            image_resized = cv2.resize(image, (64, 64))

            # 📌 Trích xuất đặc trưng HOG + Vẽ HOG Visualization
            hog_features, hog_image = hog(image_resized, visualize=True, **hog_params)
            hog_features = hog_features.flatten()

            # 📌 Đọc annotation YOLO
            annotation_path = os.path.join(annotation_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            bbox_data = []
            num_bbox = 0  # Số bbox trong ảnh

            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    num_bbox = len(lines)  # Đếm số bbox trong ảnh
                    for line in lines[:max_bbox]:
                        values = line.strip().split()
                        bbox = list(map(float, values[1:]))  # Lấy x_center, y_center, width, height
                        bbox_data.extend(bbox)

            # 📌 Padding nếu bbox < max_bbox
            if len(bbox_data) < num_bbox_features:
                bbox_data.extend([0] * (num_bbox_features - len(bbox_data)))

            bbox_data = np.array(bbox_data, dtype=np.float32)[:num_bbox_features]
            feature_vector = np.hstack((hog_features, bbox_data))

            X.append(feature_vector)
            y.append(int(lines[0].split()[0]) if len(lines) > 0 else 0)
            bbox_counts.append(num_bbox)  # Lưu số bbox của ảnh này

            # 📌 Hiển thị ảnh gốc & HOG visualization cho 1 ảnh đầu tiên
            if len(X) == 1:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image_resized, cmap='gray')
                axes[0].set_title("Ảnh sau khi resize")

                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                axes[1].imshow(hog_image_rescaled, cmap='gray')
                axes[1].set_title("HOG Visualization")

                plt.show()

# 📌 Chuyển thành numpy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 📌 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Biểu đồ phân bố số lượng bounding box trên mỗi ảnh
plt.figure(figsize=(8, 5))
plt.hist(bbox_counts, bins=20, color='blue', alpha=0.7)
plt.xlabel("Số lượng bounding boxes")
plt.ylabel("Số lượng ảnh")
plt.title("Phân phối số lượng bbox trên ảnh")
plt.show()

print("📌 Kích thước feature vector:", X_scaled.shape)
