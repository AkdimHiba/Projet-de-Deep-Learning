import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from ultralytics import YOLO
from datetime import datetime

# =========================
# CHEMINS ABSOLUS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR           = os.path.join(BASE_DIR, "content")
FACE_PROTO            = os.path.join(BASE_DIR, "content", "deploy.prototxt")
FACE_MODEL_PATH       = os.path.join(BASE_DIR, "content", "res10_300x300_ssd_iter_140000.caffemodel")
FACE_LANDMARKER_MODEL = os.path.join(BASE_DIR, "content", "face_landmarker.task")
MASK_MODEL_PATH       = os.path.join(BASE_DIR, "content", "mask_model.h5")
YOLO_MODEL_PATH       = os.path.join(BASE_DIR, "content", "yolov8n.pt")
GLOVE_MODEL_PATH      = os.path.join(BASE_DIR, "content", "glove_yolov8n.pt")

os.makedirs(CONTENT_DIR, exist_ok=True)

# =========================
# AUTO-DOWNLOAD REQUIRED FILES
# =========================
if not os.path.exists(FACE_PROTO):
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt",
        FACE_PROTO
    )
if not os.path.exists(FACE_MODEL_PATH):
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        FACE_MODEL_PATH
    )
if not os.path.exists(FACE_LANDMARKER_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        FACE_LANDMARKER_MODEL
    )

# =========================
# CONFIG
# =========================
MASK_INPUT_SIZE      = (224, 224)
EAR_THRESH           = 0.30
CLOSED_EYES_FRAME    = 8
PHONE_CONF_THRESHOLD = 0.35
GLOVE_CONF_THRESHOLD = 0.40
FACE_CONF_THRESHOLD  = 0.5

# =========================
# LOAD MASK MODEL
# =========================
mask_model = None
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    mask_model = load_model(MASK_MODEL_PATH, compile=False)
    print("[INFO] Modele charge via tensorflow.keras")
except Exception as e1:
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        mask_model = tf.keras.models.load_model(MASK_MODEL_PATH, compile=False)
        print("[INFO] Modele charge via tf.keras")
    except Exception as e2:
        try:
            import keras
            from keras.applications.mobilenet_v2 import preprocess_input
            mask_model = keras.saving.load_model(MASK_MODEL_PATH, compile=False)
            print("[INFO] Modele charge via keras standalone")
        except Exception as e3:
            raise RuntimeError(f"Impossible de charger mask_model.h5 : {e3}")

try:
    preprocess_input
except NameError:
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except:
        from keras.applications.mobilenet_v2 import preprocess_input

# =========================
# LOAD OTHER MODELS
# =========================
phone_model = YOLO(YOLO_MODEL_PATH)
glove_model = YOLO(GLOVE_MODEL_PATH)
face_net    = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL_PATH)

# =========================
# MEDIAPIPE FACE LANDMARKER
# =========================
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = FaceLandmarker.create_from_options(face_options)

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
closed_counter = {}

# =========================
# GLOVE DETECTION (YOLO)
# =========================
GLOVE_POSITIVE_CLASSES = {"surgical-gloves", "gloverotation"}

def run_glove_detection(frame):
    results    = glove_model(frame, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            if conf < GLOVE_CONF_THRESHOLD:
                continue
            label = str(r.names.get(cls_id, cls_id)).lower()
            if label in GLOVE_POSITIVE_CLASSES or label == "paper":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append((x1, y1, x2, y2, label, conf))
    return detections

# =========================
# HELPERS
# =========================
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def preprocess_mask_face(face_bgr):
    face = cv2.resize(face_bgr, MASK_INPUT_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32")
    face = preprocess_input(face)
    return np.expand_dims(face, axis=0)

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                  (300,300), (104., 177., 123.))
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < FACE_CONF_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w-1,x2), min(h-1,y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1,y1,x2,y2,float(conf)))
    return boxes

def predict_mask(face_bgr):
    inp  = preprocess_mask_face(face_bgr)
    pred = float(mask_model.predict(inp, verbose=0)[0][0])
    if pred < 0.5:
        return "mask", 1.0 - pred
    return "no_mask", pred

def get_face_landmarker_results(frame_bgr, timestamp_ms):
    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return face_landmarker.detect_for_video(mp_image, timestamp_ms)

def landmarks_to_pixels(face_landmarks, w, h):
    return [(int(lm.x*w), int(lm.y*h)) for lm in face_landmarks]

def compute_ear_from_landmarks(face_landmarks, w, h):
    pts       = landmarks_to_pixels(face_landmarks, w, h)
    left_eye  = [pts[i] for i in LEFT_EYE_IDX]
    right_eye = [pts[i] for i in RIGHT_EYE_IDX]
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return ear, left_eye, right_eye

def draw_eye(frame, eye_pts, color=(0,255,0)):
    for pt in eye_pts:
        cv2.circle(frame, pt, 2, color, -1)
    for i in range(len(eye_pts)-1):
        cv2.line(frame, eye_pts[i], eye_pts[i+1], color, 1)
    cv2.line(frame, eye_pts[-1], eye_pts[0], color, 1)

def run_phone_detection(frame):
    results    = phone_model(frame, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            if conf < PHONE_CONF_THRESHOLD:
                continue
            label = str(r.names.get(cls_id, cls_id)).lower()
            if label == "cell phone":
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                detections.append((x1,y1,x2,y2,label,conf))
    return detections

def compliance_score(mask_label, drowsy, phone_detected, gloves_detected):
    score = 100
    if mask_label == "no_mask":
        score -= 40
    if not gloves_detected:
        score -= 20
    if drowsy:
        score -= 20
    if phone_detected:
        score -= 20
    return max(score, 0)

# =========================
# COMPACT HUD (coin bas-gauche)
# =========================
def draw_hud(output, mask_label, gloves_detected, drowsy, phone_detected, score, ear):
    H, W = output.shape[:2]

    # On affiche uniquement ce qui ne va pas (rouge) + le score
    alerts = []
    if mask_label != "mask":
        alerts.append("No Mask")
    if not gloves_detected:
        alerts.append("No Gloves")
    if phone_detected:
        alerts.append("Phone Detected")
    if drowsy:
        alerts.append("Drowsiness!")

    lh  = 26
    pad = 10
    bw  = 210
    bh  = pad + max(len(alerts), 1) * lh + 20 + pad
    bx  = pad
    by  = H - bh - pad

    # Fond semi-transparent
    overlay = output.copy()
    cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
    cv2.rectangle(output, (bx, by), (bx+bw, by+bh), (80,80,80), 1)

    y = by + pad + 16

    if not alerts:
        cv2.circle(output, (bx+14, y-5), 6, (0,200,80), -1)
        cv2.putText(output, "All Good!",
                    (bx+28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,200,80), 2, cv2.LINE_AA)
        y += lh
    else:
        for msg in alerts:
            cv2.circle(output, (bx+14, y-5), 6, (0,60,255), -1)
            cv2.putText(output, msg,
                        (bx+28, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0,60,255), 2, cv2.LINE_AA)
            y += lh

    # Séparateur
    cv2.line(output, (bx+8, y-4), (bx+bw-8, y-4), (70,70,70), 1)
    y += 6

    # Score global
    score_color = (0,200,80) if score >= 80 else (0,165,255) if score >= 50 else (0,60,255)
    cv2.putText(output, f"Score: {score}/100",
                (bx+12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                score_color, 2, cv2.LINE_AA)

    # Barre score sous le panneau
    bar_x = bx
    bar_y = by + bh + 2
    bar_w = bw
    bar_h = 5
    cv2.rectangle(output, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    filled = int(bar_w * score / 100)
    cv2.rectangle(output, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), score_color, -1)


# =========================
# MAIN LOOP
# =========================
print(f"[INFO] Output shape masque : {mask_model.output_shape}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    output = frame.copy()
    H, W   = frame.shape[:2]

    timestamp_ms = int((time.time() - start_time) * 1000)

    # --- Detections ---
    face_boxes     = detect_faces_dnn(frame)
    mesh_results   = get_face_landmarker_results(frame, timestamp_ms)
    face_mesh_list = mesh_results.face_landmarks if mesh_results.face_landmarks else []

    phone_dets            = run_phone_detection(frame)
    phone_detected_global = len(phone_dets) > 0

    glove_dets = run_glove_detection(frame)
    gloves_detected_global = any(
        label in GLOVE_POSITIVE_CLASSES for (_,_,_,_,label,_) in glove_dets
    )

    # --- Phone boxes ---
    for (px1,py1,px2,py2,plabel,pconf) in phone_dets:
        cv2.rectangle(output, (px1,py1), (px2,py2), (0,0,255), 2)
        cv2.putText(output, f"Tel: {pconf:.2f}",
                    (px1, max(py1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    # --- Glove boxes ---
    for (gx1,gy1,gx2,gy2,glabel,gconf) in glove_dets:
        color = (0,200,80) if glabel in GLOVE_POSITIVE_CLASSES else (0,165,255)
        cv2.rectangle(output, (gx1,gy1), (gx2,gy2), color, 2)
        cv2.putText(output, f"Gants: {gconf:.2f}",
                    (gx1, max(gy1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # --- Drowsiness ---
    drowsy_flags = []
    global_ear   = 0.0
    for idx, face_landmarks in enumerate(face_mesh_list):
        ear, left_eye, right_eye = compute_ear_from_landmarks(face_landmarks, W, H)
        draw_eye(output, left_eye)
        draw_eye(output, right_eye)
        if idx not in closed_counter:
            closed_counter[idx] = 0
        closed_counter[idx] = closed_counter[idx]+1 if ear < EAR_THRESH else 0
        is_drowsy = closed_counter[idx] >= CLOSED_EYES_FRAME
        drowsy_flags.append((ear, is_drowsy))
        if idx == 0:
            global_ear = ear
        anchor = left_eye[0]
        cv2.putText(output, f"EAR: {ear:.2f}",
                    (anchor[0], max(anchor[1]-15,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,0), 1)
        if is_drowsy:
            cv2.putText(output, "DROWSY!",
                        (anchor[0], max(anchor[1]-35,20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # --- Mask per face ---
    global_mask_label = "no_mask"
    global_drowsy     = False

    for i, (x1,y1,x2,y2,fconf) in enumerate(face_boxes):
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        mask_label, mask_conf = predict_mask(face_crop)
        if mask_label == "mask":
            global_mask_label = "mask"
        _, is_drowsy = drowsy_flags[i] if i < len(drowsy_flags) else (None, False)
        if is_drowsy:
            global_drowsy = True

        col = (0,200,80) if mask_label == "mask" else (0,60,255)
        cv2.rectangle(output, (x1,y1), (x2,y2), col, 2)
        cv2.putText(output, f"{mask_label} {mask_conf:.2f}",
                    (x1, max(y1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    # --- Score global ---
    global_score = compliance_score(
        global_mask_label, global_drowsy,
        phone_detected_global, gloves_detected_global
    )

    # --- HUD compact ---
    draw_hud(output, global_mask_label, gloves_detected_global,
             global_drowsy, phone_detected_global, global_score, global_ear)

    cv2.imshow("Healthcare Protocol Surveillance", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()