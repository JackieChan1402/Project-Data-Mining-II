import cv2
import matplotlib.pyplot as plt


def draw_bounding_boxes(image_path, annotation_path, image_size):
    """
    Draws bounding boxes on an image using YOLO format annotations.

    :param image_path: Path to the image file.
    :param annotation_path: Path to the annotation text file.
    :param image_size: Tuple (width, height) of the original image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    height, width = image.shape[:2]  # Automatically get image dimensions

    # Read annotations
    try:
        with open(annotation_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}")
        return

    for line in lines:
        values = line.strip().split()
        if len(values) != 5:
            print(f"Warning: Incorrect format in annotation file: {line}")
            continue

        class_id = int(values[0])
        x_center, y_center, w, h = map(float, values[1:])

        # Convert normalized coordinates to pixel values
        x_center, y_center = int(x_center * width), int(y_center * height)
        w, h = int(w * width), int(h * height)

        # Calculate top-left and bottom-right corners
        x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
        x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)

        # Draw bounding box
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for class 0, Blue for class 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Class {class_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Convert BGR to RGB for correct display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image with bounding boxes
    plt.figure(figsize=(20, 10))  # Set a reasonable display size
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# Example usage
image_path = "PART_1/images/a_101.jpg"  # Change this if needed
annotation_path = "PART_1/6categories/a_101.txt"

draw_bounding_boxes(image_path, annotation_path, None)
