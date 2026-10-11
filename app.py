import os
import io
import cv2
import json
import time
import base64
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

# =========================================================
# CONFIGURATION & MODEL LOADING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fish_knn_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
VIDEO_PATH = os.path.join(BASE_DIR, "marine_video.mp4")
IMAGE_SIZE = (64, 64)

# Load machine learning model & encoder
print("[ReefLens] Loading KNN model and label encoder...")
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    CLASSES = list(encoder.classes_)
    print(f"[ReefLens] Loaded successfully! {len(CLASSES)} classes: {CLASSES}")
except Exception as e:
    print(f"[ReefLens] Error loading model: {e}")
    model = None
    encoder = None
    CLASSES = []

# =========================================================
# SPECIES ENCYCLOPEDIA / METADATA
# =========================================================

SPECIES_INFO = {
    "Puffers": {
        "scientific_name": "Tetraodontidae",
        "description": "Known for inflating rapidly to deter predators using water or air. Highly toxic to predators yet graceful reef dwellers.",
        "habitat": "Coral reefs, lagoons & tropical coastal waters",
        "depth": "1 - 100 meters",
        "diet": "Invertebrates, algae, mollusks & crustaceans",
        "status": "Vulnerable / Least Concern (species-dependent)",
        "badge_color": "#ffbe0b"
    },
    "Sea Rays": {
        "scientific_name": "Batoidea",
        "description": "Cartilaginous fish distinguished by flat bodies and enlarged pectoral fins fused to the head, gliding seamlessly through waters.",
        "habitat": "Sandy bottoms, coral fringes & open open pelagic ocean",
        "depth": "0 - 1,000+ meters",
        "diet": "Mollusks, crustaceans, small benthic fish",
        "status": "Vulnerable (many stingray & manta species)",
        "badge_color": "#3a86ff"
    },
    "Sea Urchins": {
        "scientific_name": "Echinoidea",
        "description": "Globular spiny echinoderms playing a vital role in balancing reef ecosystems by grazing on macroalgae.",
        "habitat": "Rocky shorelines, coral reef crevices & kelp beds",
        "depth": "Intertidal to 5,000 meters",
        "diet": "Algae, bryozoans, decaying organic matter",
        "status": "Ecologically critical keystone grazer",
        "badge_color": "#8338ec"
    },
    "Seahorse": {
        "scientific_name": "Hippocampus",
        "description": "Unique syngnathid fish with an upright posture, prehensile tail, coronet, and male pregnancy.",
        "habitat": "Seagrass meadows, mangrove estuaries & shallow reefs",
        "depth": "0.5 - 30 meters",
        "diet": "Microscopic crustaceans & larval invertebrates",
        "status": "Vulnerable (CITES protected)",
        "badge_color": "#06d6a0"
    },
    "Seal": {
        "scientific_name": "Pinnipedia",
        "description": "Sleek, semi-aquatic marine mammals perfectly adapted to cold ocean currents with thick blubber layers.",
        "habitat": "Coastal margins, ice floes & subpolar waters",
        "depth": "Surface to 600 meters",
        "diet": "Fish, squid, mollusks & crustaceans",
        "status": "Protected marine mammal",
        "badge_color": "#4cc9f0"
    },
    "Sharks": {
        "scientific_name": "Selachimorpha",
        "description": "Apex ocean predators with cartilaginous skeletons and acute sensory systems critical for ecosystem health.",
        "habitat": "Global marine biomes: reefs, pelagic & deep sea",
        "depth": "0 - 2,000 meters",
        "diet": "Fish, marine mammals, crustaceans, cephalopods",
        "status": "Vulnerable / Endangered (many species)",
        "badge_color": "#ef476f"
    },
    "Shrimp": {
        "scientific_name": "Caridea / Dendrobranchiata",
        "description": "Vital decapod crustaceans serving essential roles in reef health, parasite cleaning stations, and food webs.",
        "habitat": "Reef crevices, sponges, sea anemones & sand flats",
        "depth": "0 - 5,000 meters",
        "diet": "Detritus, plankton & reef parasites",
        "status": "Abundant keystone prey item",
        "badge_color": "#fb5607"
    },
    "Squid": {
        "scientific_name": "Teuthida",
        "description": "Intelligent, fast-swimming cephalopods featuring eight arms, two tentacles, jet propulsion, and chromatophore camouflage.",
        "habitat": "Open pelagic ocean, reef slopes & abyssal zones",
        "depth": "0 - 1,000+ meters",
        "diet": "Fish, crustaceans & other cephalopods",
        "status": "Least Concern (rapid generational lifecycle)",
        "badge_color": "#b5179e"
    },
    "Starfish": {
        "scientific_name": "Asteroidea",
        "description": "Iconic star-shaped marine invertebrates using hydraulic water vascular systems and tube feet for locomotion.",
        "habitat": "Tide pools, coral reefs, sandy substrate & deep trenches",
        "depth": "Intertidal to 6,000 meters",
        "diet": "Clams, oysters, sponges, coral polyps & snails",
        "status": "Ecologically sensitive bioindicator",
        "badge_color": "#ff006e"
    },
    "Turtle_Tortoise": {
        "scientific_name": "Chelonioidea",
        "description": "Ancient marine reptiles navigating thousands of nautical miles using Earth's geomagnetic field.",
        "habitat": "Tropical and subtropical nesting beaches & feeding reefs",
        "depth": "0 - 290 meters",
        "diet": "Seagrass, sponges, jellyfish & tunicates",
        "status": "Endangered / Critically Endangered (IUCN)",
        "badge_color": "#2ec4b6"
    },
    "Whale": {
        "scientific_name": "Cetacea",
        "description": "Majestic cetaceans spanning gentle filter-feeding giants to sophisticated acoustic pod hunters.",
        "habitat": "Open ocean migratory corridors & polar feeding grounds",
        "depth": "Surface to 3,000 meters (deep divers)",
        "diet": "Krill, schooling fish, plankton & squid",
        "status": "Vulnerable / Conservation Dependent",
        "badge_color": "#0077b6"
    }
}

# Real Evaluation Metrics (from evaluate.py test split)
EVAL_METRICS = {
    "overall_accuracy": 88.84,
    "total_test_samples": 1362,
    "total_dataset_samples": 6808,
    "classes_breakdown": {
        "Puffers": {"precision": 1.00, "recall": 0.89, "f1": 0.94, "support": 57, "dataset_count": 285},
        "Sea Rays": {"precision": 0.91, "recall": 0.84, "f1": 0.87, "support": 103, "dataset_count": 517},
        "Sea Urchins": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 116, "dataset_count": 579},
        "Seahorse": {"precision": 0.99, "recall": 0.86, "f1": 0.92, "support": 95, "dataset_count": 478},
        "Seal": {"precision": 0.76, "recall": 0.84, "f1": 0.80, "support": 83, "dataset_count": 414},
        "Sharks": {"precision": 0.83, "recall": 0.90, "f1": 0.86, "support": 118, "dataset_count": 590},
        "Shrimp": {"precision": 0.95, "recall": 0.79, "f1": 0.86, "support": 98, "dataset_count": 488},
        "Squid": {"precision": 0.85, "recall": 0.91, "f1": 0.88, "support": 97, "dataset_count": 483},
        "Starfish": {"precision": 0.93, "recall": 0.82, "f1": 0.87, "support": 100, "dataset_count": 499},
        "Turtle_Tortoise": {"precision": 0.88, "recall": 0.95, "f1": 0.91, "support": 381, "dataset_count": 1903},
        "Whale": {"precision": 0.85, "recall": 0.90, "f1": 0.88, "support": 114, "dataset_count": 572}
    }
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def preprocess_image_for_model(img_bgr):
    """Preprocess cv2 BGR image into (1, 12288) normalized feature vector"""
    resized = cv2.resize(img_bgr, IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = rgb.astype("float32") / 255.0
    feat = norm.flatten().reshape(1, -1)
    return feat, resized

def run_classification(img_bgr):
    """Run KNN classification and get probabilities + annotated image"""
    if model is None or encoder is None:
        raise ValueError("Model is not initialized.")

    features, resized_model_view = preprocess_image_for_model(img_bgr)
    
    # Predict probabilities across classes
    probabilities = model.predict_proba(features)[0]
    pred_idx = model.predict(features)[0]
    predicted_species = encoder.inverse_transform([pred_idx])[0]
    top_confidence = float(probabilities[pred_idx] * 100)

    # Rank top 5 classes
    sorted_indices = np.argsort(probabilities)[::-1]
    rankings = []
    for idx in sorted_indices[:5]:
        cls_name = encoder.classes_[idx]
        conf = float(probabilities[idx] * 100)
        rankings.append({
            "species": cls_name,
            "confidence": round(conf, 1),
            "info": SPECIES_INFO.get(cls_name, {})
        })

    # Create annotated version of the image
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]
    badge_text = f"{predicted_species} ({top_confidence:.1f}%)"
    
    # Draw top banner box
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 20, 35), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.line(annotated, (0, 55), (w, 55), (0, 245, 212), 2)
    
    # Text
    cv2.putText(
        annotated,
        badge_text,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 245, 212),
        2,
        cv2.LINE_AA
    )

    # Encode images to base64
    _, annot_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    annot_b64 = "data:image/jpeg;base64," + base64.b64encode(annot_buf).decode("utf-8")

    _, model_buf = cv2.imencode(".jpg", resized_model_view, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    model_b64 = "data:image/jpeg;base64," + base64.b64encode(model_buf).decode("utf-8")

    info = SPECIES_INFO.get(predicted_species, {})

    return {
        "species": predicted_species,
        "confidence": round(top_confidence, 1),
        "rankings": rankings,
        "annotated_image": annot_b64,
        "model_input_image": model_b64,
        "species_info": info
    }

# =========================================================
# SIFT TEMPLATE PREPARATION (Cached for SIFT Engine)
# =========================================================

sift_templates = []
sift_initialized = False

def init_sift_templates():
    global sift_templates, sift_initialized
    if sift_initialized or not os.path.exists(DATASET_PATH):
        return
    try:
        sift = cv2.SIFT_create(nfeatures=400)
        print("[ReefLens] Pre-indexing SIFT templates for video matcher...")
        for class_name in sorted(os.listdir(DATASET_PATH)):
            class_path = os.path.join(DATASET_PATH, class_name)
            if not os.path.isdir(class_path):
                continue
            count = 0
            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(class_path, file))
                    if img is None:
                        continue
                    img = cv2.resize(img, (240, 240))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kp, des = sift.detectAndCompute(gray, None)
                    if des is not None:
                        sift_templates.append({
                            "class": class_name,
                            "descriptors": des
                        })
                        count += 1
                        if count >= 6: # Fast subset per class for live stream speed
                            break
        sift_initialized = True
        print(f"[ReefLens] Indexed {len(sift_templates)} SIFT templates.")
    except Exception as e:
        print(f"[ReefLens] SIFT initialization notice: {e}")

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "classes_count": len(CLASSES),
        "classes": CLASSES,
        "accuracy": EVAL_METRICS["overall_accuracy"],
        "dataset_total": EVAL_METRICS["total_dataset_samples"]
    })

@app.route("/api/samples")
def get_samples():
    """Return 1 or 2 sample images per class for instant testing"""
    samples = []
    if os.path.exists(DATASET_PATH):
        for cls_name in sorted(os.listdir(DATASET_PATH)):
            cls_dir = os.path.join(DATASET_PATH, cls_name)
            if os.path.isdir(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if files:
                    samples.append({
                        "species": cls_name,
                        "filename": files[0],
                        "url": f"/api/sample-image/{cls_name}/{files[0]}",
                        "total_images": len(files)
                    })
    return jsonify({"samples": samples})

@app.route("/api/sample-image/<species>/<filename>")
def serve_sample_image(species, filename):
    folder = os.path.join(DATASET_PATH, species)
    return send_from_directory(folder, filename)

@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        # Check if sample path was provided
        if "sample_species" in request.form and "sample_filename" in request.form:
            sp = request.form["sample_species"]
            fn = request.form["sample_filename"]
            img_path = os.path.join(DATASET_PATH, sp, fn)
            if not os.path.exists(img_path):
                return jsonify({"error": "Sample image not found"}), 404
            img_bgr = cv2.imread(img_path)
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file uploaded"}), 400
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif request.is_json and "image_b64" in request.json:
            b64_str = request.json["image_b64"].split(",")[-1]
            file_bytes = np.frombuffer(base64.b64decode(b64_str), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "No valid image provided"}), 400

        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        result = run_classification(img_bgr)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/metrics")
def get_metrics():
    return jsonify(EVAL_METRICS)

# =========================================================
# VIDEO STREAMING GENERATOR
# =========================================================

def generate_video_stream(engine="knn"):
    """MJPEG stream with real-time detection boxes"""
    if not os.path.exists(VIDEO_PATH):
        # Return fallback placeholder
        while True:
            blank = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(blank, "marine_video.mp4 not found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_counter += 1
        display_frame = cv2.resize(frame, (800, 450))
        active_detections = 0

        if engine == "knn" and model is not None:
            # Motion detection + KNN classification
            mask = bg_subtractor.apply(display_frame)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < 1000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w < 30 or h < 30:
                    continue

                fish_crop = display_frame[y:y+h, x:x+w]
                if fish_crop.size == 0:
                    continue

                # Run KNN
                resized = cv2.resize(fish_crop, IMAGE_SIZE)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                feat = rgb.flatten().reshape(1, -1)
                pred = model.predict(feat)[0]
                proba = model.predict_proba(feat)[0][pred] * 100
                fish_name = encoder.inverse_transform([pred])[0]

                # Neon box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 245, 212), 2)
                # Label box
                label = f"{fish_name} {proba:.0f}%"
                cv2.rectangle(display_frame, (x, max(y - 24, 0)), (x + len(label) * 11 + 8, max(y, 24)), (12, 22, 38), -1)
                cv2.putText(
                    display_frame,
                    label,
                    (x + 4, max(y - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 245, 212),
                    1,
                    cv2.LINE_AA
                )
                active_detections += 1

        # Draw Telemetry HUD
        hud_text = f"ReefLens Engine: {engine.upper()} | Frame: {frame_counter} | Detections: {active_detections}"
        cv2.putText(display_frame, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        # Pace to ~24-30 fps
        time.sleep(0.033)

    cap.release()

@app.route("/api/video-stream")
def video_stream():
    engine = request.args.get("engine", "knn")
    return Response(
        generate_video_stream(engine),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n=======================================================")
    print(f"🌊 ReefLens Local Web App running at:")
    print(f"👉 http://127.0.0.1:{port}")
    print(f"👉 http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
