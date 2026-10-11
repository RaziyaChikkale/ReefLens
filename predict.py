import cv2
import joblib
import numpy as np


IMAGE_SIZE = (64, 64)

MODEL_PATH = "models/fish_knn_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# ==============================
# ENTER IMAGE PATH
# ==============================

image_path = input(
    "Enter fish image path: "
)


# ==============================
# READ IMAGE
# ==============================

image = cv2.imread(image_path)

if image is None:

    print("ERROR: Image not found!")

    exit()


original = image.copy()


# ==============================
# PREPROCESS
# ==============================

image = cv2.resize(
    image,
    IMAGE_SIZE
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = image.astype(
    "float32"
) / 255.0

image = image.flatten()

image = image.reshape(
    1,
    -1
)


# ==============================
# PREDICT
# ==============================

prediction = model.predict(image)[0]

fish_name = encoder.inverse_transform(
    [prediction]
)[0]


# ==============================
# DISPLAY
# ==============================

print("\nPredicted Fish:", fish_name)

cv2.putText(
    original,
    f"Fish: {fish_name}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow(
    "Fish Classification",
    original
)

cv2.waitKey(0)

cv2.destroyAllWindows()