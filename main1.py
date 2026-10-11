import cv2
import os
import numpy as np

# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "dataset"
VIDEO_PATH = "marine_video.mp4"

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 450

# Feature matching settings
MIN_MATCHES = 8
RATIO_TEST = 0.75

# Process every 2nd frame
FRAME_SKIP = 2

# =========================================================
# CREATE SIFT
# =========================================================

print("Starting ReefLens...")

sift = cv2.SIFT_create(
    nfeatures=500
)

# FLANN matcher
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

# =========================================================
# LOAD DATASET IMAGES
# =========================================================

print("\nLoading dataset...")

templates = []

if not os.path.exists(DATASET_PATH):

    print("ERROR: dataset folder not found!")

    exit()

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    print("Loading:", class_name)

    image_count = 0

    for file in os.listdir(class_path):

        if not file.lower().endswith(
            (".jpg", ".jpeg")
        ):
            continue

        image_path = os.path.join(
            class_path,
            file
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize template
        image = cv2.resize(
            image,
            (300, 300)
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Detect SIFT features
        keypoints, descriptors = sift.detectAndCompute(
            gray,
            None
        )

        if descriptors is None:
            continue

        templates.append(
            {
                "class": class_name,
                "keypoints": keypoints,
                "descriptors": descriptors
            }
        )

        image_count += 1

        # Only use 20 templates per class
        if image_count >= 20:
            break

print("\nTemplates loaded:", len(templates))

if len(templates) == 0:

    print("ERROR: No usable JPG images found.")
    exit()

# =========================================================
# OPEN VIDEO
# =========================================================

print("\nOpening video...")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("ERROR: Could not open marine_video.mp4")

    exit()

# =========================================================
# VIDEO LOOP
# =========================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number % FRAME_SKIP != 0:
        continue

    # Resize
    frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect features in video frame
    frame_keypoints, frame_descriptors = sift.detectAndCompute(
        gray_frame,
        None
    )

    if frame_descriptors is None:

        cv2.imshow(
            "ReefLens - Marine Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    detections = []

    # =====================================================
    # COMPARE VIDEO WITH DATASET
    # =====================================================

    for template in templates:

        try:

            matches = flann.knnMatch(
                template["descriptors"],
                frame_descriptors,
                k=2
            )

        except cv2.error:
            continue

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < RATIO_TEST * n.distance:

                good_matches.append(m)

        # =================================================
        # ENOUGH MATCHES?
        # =================================================

        if len(good_matches) < MIN_MATCHES:
            continue

        # Get matched points
        points = []

        for match in good_matches:

            x, y = frame_keypoints[
                match.trainIdx
            ].pt

            points.append(
                [x, y]
            )

        points = np.array(
            points,
            dtype=np.float32
        )

        # =================================================
        # CREATE BOX AROUND MATCHED REGION
        # =================================================

        x_min = int(
            np.min(points[:, 0])
        )

        y_min = int(
            np.min(points[:, 1])
        )

        x_max = int(
            np.max(points[:, 0])
        )

        y_max = int(
            np.max(points[:, 1])
        )

        width = x_max - x_min
        height = y_max - y_min

        # Ignore tiny boxes
        if width < 25 or height < 25:
            continue

        # Add some padding
        padding = 15

        x_min = max(
            0,
            x_min - padding
        )

        y_min = max(
            0,
            y_min - padding
        )

        x_max = min(
            DISPLAY_WIDTH - 1,
            x_max + padding
        )

        y_max = min(
            DISPLAY_HEIGHT - 1,
            y_max + padding
        )

        detections.append(
            {
                "class": template["class"],
                "x1": x_min,
                "y1": y_min,
                "x2": x_max,
                "y2": y_max,
                "matches": len(good_matches)
            }
        )

    # =====================================================
    # REMOVE DUPLICATE DETECTIONS
    # =====================================================

    # Keep strongest matches
    detections.sort(
        key=lambda x: x["matches"],
        reverse=True
    )

    final_detections = []

    for detection in detections:

        keep = True

        for existing in final_detections:

            # Calculate overlap
            x1 = max(
                detection["x1"],
                existing["x1"]
            )

            y1 = max(
                detection["y1"],
                existing["y1"]
            )

            x2 = min(
                detection["x2"],
                existing["x2"]
            )

            y2 = min(
                detection["y2"],
                existing["y2"]
            )

            if x2 > x1 and y2 > y1:

                intersection = (
                    (x2 - x1) *
                    (y2 - y1)
                )

                area1 = (
                    (detection["x2"] -
                     detection["x1"]) *
                    (detection["y2"] -
                     detection["y1"])
                )

                area2 = (
                    (existing["x2"] -
                     existing["x1"]) *
                    (existing["y2"] -
                     existing["y1"])
                )

                union = area1 + area2 - intersection

                iou = intersection / union

                if iou > 0.5:

                    keep = False
                    break

        if keep:

            final_detections.append(
                detection
            )

    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    for detection in final_detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        class_name = detection["class"]

        matches = detection["matches"]

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} ({matches})"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "ReefLens - Marine Species Detection",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("\nReefLens finished.")