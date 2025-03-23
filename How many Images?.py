import os
import cv2
from PIL import Image


def get_image_info(folder_path, extensions=['.jpg', '.png', '.jpeg', '.bmp', '.gif']):
    """
    Extracts detailed information about images in a folder.

    :param folder_path: Path to the folder containing images.
    :param extensions: List of image file extensions to check.
    :return: None (Prints results).
    """
    total_images = 0
    sizes = []
    resolutions = []
    color_modes = {}

    for file in os.listdir(folder_path):
        if file.lower().endswith(tuple(extensions)):  # Check valid image extensions
            total_images += 1
            file_path = os.path.join(folder_path, file)

            # Get file size in KB
            file_size = os.path.getsize(file_path) / 1024  # Convert bytes to KB
            sizes.append(file_size)

            # Get image resolution and color mode
            img = cv2.imread(file_path)
            if img is not None:
                h, w, c = img.shape
                resolutions.append((w, h))

                # Detect grayscale or color
                pil_img = Image.open(file_path)
                color_mode = "Grayscale" if pil_img.mode == "L" else "Color"
                color_modes[file] = color_mode

            # Get image format
            img_format = file.split('.')[-1].upper()

            # Print info for each image
            print(f"📸 {file}:")
            print(f"   - Resolution: {w}x{h} pixels")
            print(f"   - File Size: {file_size:.2f} KB")
            print(f"   - Format: {img_format}")
            print(f"   - Color Mode: {color_mode}\n")

    # Print overall statistics
    print("=" * 40)
    print(f"📌 Total images: {total_images}")
    if sizes:
        print(
            f"📏 Min Size: {min(sizes):.2f} KB | Max Size: {max(sizes):.2f} KB | Avg Size: {sum(sizes) / len(sizes):.2f} KB")
    if resolutions:
        print(f"📐 Min Resolution: {min(resolutions)} (WxH) | Max Resolution: {max(resolutions)} (WxH)")


# Example usage
folder_path = "PART_3/PART_3/images"  # Replace with your folder path
get_image_info(folder_path)
