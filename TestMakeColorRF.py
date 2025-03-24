import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
rf_objects = joblib.load("rf_objects.pkl")
rf_classes = joblib.load("rf_classes.pkl")
rf_centers = joblib.load("rf_centers.pkl")
pca = joblib.load("pca_model.pkl")

# Define colors for each class
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


# Function to extract features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128)).flatten()
    return gray


# Function to predict number of objects, classes, and positions
def predict_and_display(image_path, rf_objects, rf_classes, rf_centers, pca):
    image = cv2.imread(image_path)
    h_orig, w_orig, _ = image.shape  # Get original image size

    features = extract_features(image)
    features_pca = pca.transform([features])

    predicted_objects = int(rf_objects.predict(features_pca)[0])
    predicted_classes = rf_classes.predict(features_pca)[0].astype(int)
    predicted_centers = rf_centers.predict(features_pca)[0].reshape(-1, 2)

    class_counts = {f"Class {i}": predicted_classes[i] for i in range(len(predicted_classes)) if
                    predicted_classes[i] > 0}

    # Draw predicted object centers (adjusting coordinates to original size)
    for class_id, count in enumerate(predicted_classes):
        color = COLORS[class_id % len(COLORS)]  # Assign color based on class
        for i in range(count):
            x, y = predicted_centers[i]
            if x > 0 and y > 0:  # Ignore zero-padding values
                x, y = int(x * w_orig), int(y * h_orig)  # Scale coordinates back to original size
                cv2.circle(image, (x, y), 20, color, -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Objects: {predicted_objects}\nClasses: {class_counts}")
    plt.axis("off")
    plt.show()


# Test with an image
test_image_path = "a_1013.jpg"
predict_and_display(test_image_path, rf_objects, rf_classes, rf_centers, pca)
