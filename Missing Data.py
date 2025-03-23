import os
import cv2


def check_missing_data(image_folder, annotation_folder):
    """
    Checks for missing or corrupt images and annotations.

    :param image_folder: Path to the folder containing images.
    :param annotation_folder: Path to the folder containing YOLO annotation text files.
    """
    missing_annotations = []
    missing_images = []
    corrupt_images = []

    # Get all image filenames (without extension)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
    annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotation_folder) if f.endswith('.txt')}

    # Check for missing annotation files
    for img in image_files:
        if img not in annotation_files:
            missing_annotations.append(img)

    # Check for missing images
    for ann in annotation_files:
        if ann not in image_files:
            missing_images.append(ann)

    # Check for corrupt images
    for img in image_files:
        image_path = os.path.join(image_folder, img + ".jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, img + ".png")  # Try PNG if JPG not found

        image = cv2.imread(image_path)
        if image is None:
            corrupt_images.append(img)

    # Print results
    print("=== Missing Data Report ===")
    print(f"Total images: {len(image_files)}")
    print(f"Total annotations: {len(annotation_files)}")
    print(f"Missing annotations: {len(missing_annotations)} -> {missing_annotations}")
    print(f"Missing images: {len(missing_images)} -> {missing_images}")
    print(f"Corrupt images: {len(corrupt_images)} -> {corrupt_images}")

    return missing_annotations, missing_images, corrupt_images


# Example usage
image_folder = "PART_1/images"  # Change to your dataset image folder
annotation_folder = "PART_1/6categories"  # Change to YOLO annotation folder

check_missing_data(image_folder, annotation_folder)
