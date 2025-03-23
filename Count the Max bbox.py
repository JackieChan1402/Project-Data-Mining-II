import os

# 📌 Thư mục chứa annotation YOLO
annotation_folder = "PART_1/6categories"  # Thay bằng thư mục thực tế

max_bbox_count = 0  # Số bbox lớn nhất
bbox_counts = []  # Danh sách lưu số bbox của từng file

# 📌 Duyệt qua tất cả file annotation trong thư mục
for annotation_file in os.listdir(annotation_folder):
    if annotation_file.endswith(".txt"):
        annotation_path = os.path.join(annotation_folder, annotation_file)

        with open(annotation_path, 'r') as f:
            bbox_count = len(f.readlines())  # Đếm số dòng = số bbox
            bbox_counts.append(bbox_count)

            if bbox_count > max_bbox_count:
                max_bbox_count = bbox_count

# 📌 Kết quả
print(f"📌 Số bbox lớn nhất trong tập dữ liệu: {max_bbox_count}")
print(f"📌 Phân bố số bbox (min: {min(bbox_counts)}, max: {max_bbox_count}, trung bình: {sum(bbox_counts)/len(bbox_counts):.2f})")
