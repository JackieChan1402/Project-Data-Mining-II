import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 📌 Đọc ảnh đầu vào
image_path = "PART_1/images/a_101.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 📌 Trích xuất đặc trưng SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 📌 Áp dụng K-Means Clustering
K = 3  # Số cụm (tùy chọn)
kmeans = KMeans(n_clusters=K, random_state=42, n_init=1000)
labels = kmeans.fit_predict(descriptors)

# 📌 Màu sắc đại diện cho từng cụm
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Đỏ, Xanh lá, Xanh dương

# 📌 Vẽ keypoints với màu theo cụm
clustered_image = image.copy()
for i, kp in enumerate(keypoints):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cluster_color = colors[labels[i] % len(colors)]
    cv2.circle(clustered_image, (x, y), 5, cluster_color, -1)

# 📌 Hiển thị kết quả
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Ảnh Gốc")

plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
plt.title(f"Phân Loại Vật Thể (K={K})")

plt.show()
