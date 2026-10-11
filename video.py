import cv2
import joblib
import numpy as np


# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (64, 64)

MODEL_PATH = "models/fish_knn_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

VIDEO_PATH = "marine_video.mp4"


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load(MODEL_PATH)

encoder = joblib.load(
    ENCODER_PATH
)


# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


# ==============================
# BACKGROUND SUBTRACTOR
# ==============================

background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=False
)


print("Starting fish detection...")
print("Press Q to stop.")


# ==============================
# VIDEO LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ==========================
    # CREATE MOTION MASK
    # ==========================

    mask = background.apply(
        frame
    )


    # ==========================
    # REMOVE NOISE
    # ==========================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ==========================
    # FIND CONTOURS
    # ==========================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # Ignore tiny objects
        if area < 1000:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Ignore extremely small boxes
        if w < 30 or h < 30:
            continue


        # ======================
        # CROP FISH
        # ======================

        fish = frame[
            y:y+h,
            x:x+w
        ]


        if fish.size == 0:
            continue


        # ======================
        # PREPROCESS FISH
        # ======================

        resized = cv2.resize(
            fish,
            IMAGE_SIZE
        )

        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        resized = resized.astype(
            "float32"
        ) / 255.0

        features = resized.flatten()

        features = features.reshape(
            1,
            -1
        )


        # ======================
        # PREDICT FISH TYPE
        # ======================

        prediction = model.predict(
            features
        )[0]

        fish_name = encoder.inverse_transform(
            [prediction]
        )[0]


        # ======================
        # DRAW BOUNDING BOX
        # ======================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        # ======================
        # DISPLAY LABEL
        # ======================

        label = f"Fish: {fish_name}"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ==========================
    # DISPLAY VIDEO
    # ==========================

    cv2.imshow(
        "Fish Detection and Classification",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# ==============================
# RELEASE
# ==============================

cap.release()

cv2.destroyAllWindows()

print("Video processing finished.")