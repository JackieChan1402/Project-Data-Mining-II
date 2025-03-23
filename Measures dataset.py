import os
import numpy as np
import pandas as pd


def load_yolo_data(annotation_folder):
    """
    Loads bounding box data from YOLO annotation files.
    :param annotation_folder: Path to the folder containing YOLO .txt files.
    :return: DataFrame with columns ['class_id', 'x_center', 'y_center', 'width', 'height']
    """
    data = []
    for file in os.listdir(annotation_folder):
        if file.endswith('.txt'):
            with open(os.path.join(annotation_folder, file), 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id, x, y, w, h = map(float, values)
                        data.append([int(class_id), x, y, w, h])

    return pd.DataFrame(data, columns=['class_id', 'x_center', 'y_center', 'width', 'height'])


def compute_statistics(df):
    """
    Computes mean, variance, covariance, and correlation for numerical features.
    """
    print("=== Statistical Measures ===")
    print("Mean:\n", df.mean())
    print("\nVariance:\n", df.var())
    print("\nCovariance:\n", df.cov())
    print("\nCorrelation:\n", df.corr())

    # Handling categorical feature (class_id)
    print("\nClass Distribution:\n", df['class_id'].value_counts())


def main():
    annotation_folder = "PART_1/6categories"  # Change to dataset folder
    df = load_yolo_data(annotation_folder)

    if df.empty:
        print("No data found!")
    else:
        compute_statistics(df)


if __name__ == "__main__":
    main()
