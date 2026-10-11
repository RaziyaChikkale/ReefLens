import cv2
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)
MODEL_PATH = "models/fish_knn_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# ==============================
# CREATE MODEL FOLDER
# ==============================

os.makedirs("models", exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================

X = []
y = []

print("\nLoading dataset...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(root, file)

        # Folder name = fish type
        label = os.path.basename(root)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(image, IMAGE_SIZE)

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image.astype("float32") / 255.0

        # Flatten image
        image = image.flatten()

        X.append(image)
        y.append(label)


# ==============================
# CONVERT TO NUMPY
# ==============================

X = np.array(X)
y = np.array(y)

print("Images loaded:", len(X))
print("Image feature size:", X.shape)


# ==============================
# ENCODE LABELS
# ==============================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nFish classes:")

for i, name in enumerate(encoder.classes_):
    print(i, "=", name)


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ==============================
# CREATE KNN
# ==============================

print("\nTraining KNN model...")

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    n_jobs=-1
)


# ==============================
# TRAIN
# ==============================

model.fit(X_train, y_train)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)


print("\n================================")
print("MODEL TRAINING COMPLETE")
print("================================")

print("Model saved:", MODEL_PATH)
print("Encoder saved:", ENCODER_PATH)