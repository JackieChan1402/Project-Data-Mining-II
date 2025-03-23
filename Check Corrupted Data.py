import os
import cv2


def check_corrupted_data(image_folder, annotation_folder):
    """
    Checks for incorrect or corrupted data in images and annotations.

    :param image_folder: Path to the folder containing images.
    :param annotation_folder: Path to the folder containing YOLO annotation text files.
    """
    corrupt_images = []
    incorrect_annotations = []

    # Get all image filenames (without extension)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
    annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotation_folder) if f.endswith('.txt')}

    # Check for corrupt images
    for img in image_files:
        image_path = os.path.join(image_folder, img + ".jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, img + ".png")  # Try PNG if JPG not found

        image = cv2.imread(image_path)
        if image is None:
            corrupt_images.append(img)

    # Check for incorrect annotations
    for ann in annotation_files:
        annotation_path = os.path.join(annotation_folder, ann + ".txt")
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.strip().split()
                if len(values) != 5:
                    incorrect_annotations.append(ann)
                    break  # No need to check further if already incorrect

                # Check if bounding box values are within range [0,1]
                try:
                    class_id, x_center, y_center, width, height = map(float, values)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        incorrect_annotations.append(ann)
                        break
                except ValueError:
                    incorrect_annotations.append(ann)
                    break

    # Print results
    print("=== Data Integrity Report ===")
    print(f"Total images: {len(image_files)}")
    print(f"Total annotations: {len(annotation_files)}")
    print(f"Corrupt images: {len(corrupt_images)} -> {corrupt_images}")
    print(f"Incorrect annotations: {len(incorrect_annotations)} -> {incorrect_annotations}")

    return corrupt_images, incorrect_annotations


# Example usage
image_folder = "PART_1/images"  # Change to your dataset image folder
annotation_folder = "PART_1/6categories"  # Change to YOLO annotation folder

check_corrupted_data(image_folder, annotation_folder)
