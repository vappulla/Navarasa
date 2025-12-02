# real_time_emotion_deit_mediapipe_aggregated_bboxes.py
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from moviepy import VideoFileClip, ImageSequenceClip
import mediapipe as mp
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch16_224

# ------------------ USER SETTINGS ------------------
video_path = "./Navarasa/Video Project 1.mp4"                
best_model_path = "./deit_color_checkpoints/deit_color_best.pth"  
output_video_path = "./Navarasa_output/Video Project 1.mp4" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ CLASS NAMES & COLORS ------------------
class_names = [
    "Adbhutha", "Bhayaanaka", "Bheebhatsya", "Hasya",
    "Karuna", "Roudra", "Shaanta", "Shringaara", "Veera"
]

colors = [
    (255, 0, 0), (19, 69, 139), (255, 255, 0), (255, 0, 255),
    (0, 255, 0), (255, 165, 0), (0, 255, 255), (0, 0, 255), (128, 0, 128)
]

# ------------------ MODEL SETUP ------------------
img_size = 224
num_classes = len(class_names)
model = vit_small_patch16_224(num_classes=num_classes, img_size=img_size, pretrained=False)

ck = torch.load(best_model_path, map_location=device)
state = None
if isinstance(ck, dict):
    if 'model_state' in ck: state = ck['model_state']
    elif 'model_state_dict' in ck: state = ck['model_state_dict']
    else: state = ck
else:
    state = ck

# strip "module." prefix if exists
new_state = {k[len("module."):]: v if k.startswith("module.") else v for k,v in state.items()}
model.load_state_dict(new_state, strict=False)
model = model.to(device)
model.eval()
print("Loaded model weights from:", best_model_path)

# ------------------ PREPROCESS ------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ------------------ VIDEO IO ------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video opened: {video_path} | fps={fps:.2f} | size={width}x{height}")

# ------------------ MEDIAPIPE FACE DETECTION ------------------
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

def clamp_coords(x1, y1, x2, y2, w, h):
    x1i = max(0, min(w-1, int(round(x1))))
    y1i = max(0, min(h-1, int(round(y1))))
    x2i = max(0, min(w-1, int(round(x2))))
    y2i = max(0, min(h-1, int(round(y2))))
    return x1i, y1i, x2i, y2i

def infer_batch(face_pils, batch_size=16):
    if len(face_pils) == 0: return []
    tensors = [transform(im) for im in face_pils]
    x = torch.stack(tensors, dim=0).to(device)
    probs_all = []
    with torch.no_grad():
        N = x.shape[0]
        start = 0
        while start < N:
            end = min(start + batch_size, N)
            batch = x[start:end]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            start = end
    return np.vstack(probs_all)

# ------------------ FRAME PROCESSING WITH 50-FRAME AGGREGATION ------------------
FRAME_BATCH_SIZE = 50
frame_predictions = []
frame_confidences = []
annotated_frames = []
last_frames_batch = []

frame_idx = 0
t_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    face_pils = []
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = bbox.xmin * w
            y1 = bbox.ymin * h
            x2 = x1 + bbox.width * w
            y2 = y1 + bbox.height * h

            pad = 0.12
            bw = x2 - x1
            bh = y2 - y1
            x1 -= pad * bw
            y1 -= pad * bh
            x2 += pad * bw
            y2 += pad * bh

            x1i, y1i, x2i, y2i = clamp_coords(x1, y1, x2, y2, w, h)
            if x2i - x1i < 8 or y2i - y1i < 8: continue

            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0: continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            face_pils.append(pil)
            boxes.append((x1i, y1i, x2i, y2i))

    probs = infer_batch(face_pils, batch_size=16)

    # Draw bounding boxes and per-face labels
    for (x1i, y1i, x2i, y2i), prob_vec in zip(boxes, probs):
        class_idx = int(prob_vec.argmax())
        conf = float(prob_vec[class_idx])
        label = class_names[class_idx]
        color = colors[class_idx]
        # rectangle
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
        # background for text
        text = f"{label}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1i, max(0, y1i-text_h-10)), (x1i+text_w+6, y1i), color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, text, (x1i+3, y1i-6), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # Aggregate per-frame predictions (take max prob face or NoFace)
    if len(probs) > 0:
        max_idx = probs.sum(axis=0).argmax()
        conf = probs[:, max_idx].mean()
        pred_emotion = class_names[max_idx]
    else:
        pred_emotion = "NoFace"
        conf = 0.0

    frame_predictions.append(pred_emotion)
    frame_confidences.append(conf)
    last_frames_batch.append(frame.copy())

    if len(frame_predictions) == FRAME_BATCH_SIZE:
        overall_emotion = max(set(frame_predictions), key=frame_predictions.count)
        avg_conf = sum(frame_confidences)/len(frame_confidences)
        label_text = f"{overall_emotion} ({round(avg_conf*100,2)}% confidence)"
        # Add aggregated label on top-left of each frame
        for f in last_frames_batch:
            cv2.putText(f, label_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,0), 4, cv2.LINE_AA)
            annotated_frames.append(f)
        frame_predictions, frame_confidences, last_frames_batch = [], [], []

    if frame_idx % 100 == 0:
        print(f"Processed frame {frame_idx} - elapsed {time.time()-t_start:.1f}s")

# Handle leftover frames
if last_frames_batch:
    overall_emotion = max(set(frame_predictions), key=frame_predictions.count)
    avg_conf = sum(frame_confidences)/len(frame_confidences)
    label_text = f"{overall_emotion} ({round(avg_conf*100,2)}% confidence)"
    for f in last_frames_batch:
        cv2.putText(f, label_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,0), 4, cv2.LINE_AA)
        annotated_frames.append(f)

cap.release()
detector.close()

# ------------------ WRITE FINAL VIDEO ------------------
final_rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in annotated_frames]
orig_clip = VideoFileClip(video_path)
proc_clip = ImageSequenceClip(final_rgb_frames, fps=orig_clip.fps)
proc_clip = proc_clip.with_audio(orig_clip.audio)
proc_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac",
                          temp_audiofile="/tmp/temp-audio.m4a", remove_temp=True)
print("Done! Output with audio at:", output_video_path)

