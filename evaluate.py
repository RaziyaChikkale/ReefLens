import cv2
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)


# ==============================
# LOAD DATA
# ==============================

X = []
y = []

print("Loading dataset...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        path = os.path.join(root, file)

        image = cv2.imread(path)

        if image is None:
            continue

        label = os.path.basename(root)

        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = image.flatten()

        X.append(image)
        y.append(label)


X = np.array(X)
y = np.array(y)


# ==============================
# ENCODE LABELS
# ==============================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ==============================
# SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load(
    "models/fish_knn_model.pkl"
)


# ==============================
# PREDICT
# ==============================

predictions = model.predict(X_test)


# ==============================
# ACCURACY
# ==============================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)