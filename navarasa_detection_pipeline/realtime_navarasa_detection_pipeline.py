# real_time_emotion_deit_mediapipe.py
import os
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from moviepy import VideoFileClip
import mediapipe as mp
from torchvision import transforms

# ------------------ USER SETTINGS ------------------
video_path = "./Navarasa/Video Project 1.mp4"                # INPUT video (change)
best_model_path = "./deit_color_checkpoints/deit_color_best.pth"  # trained weights
temp_video_path = "/tmp/tmp_noaudio.mp4"               # intermediate no-audio video
output_video_path = "./Navarasa_output/Video Project 1.mp4" # final output with audio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# ------------------ CLASS NAMES (in index order) ------------------
# Use the same ordering your training used (from classification_report)
class_names = [
    "Adbhutha",
    "Bhayaanaka",
    "Bheebhatsya",
    "Hasya",
    "Karuna",
    "Roudra",
    "Shaanta",
    "Shringaara",
    "Veera"
]

# Colors mapping (BGR) for each class index 0..8 (as requested)
colors = [
    (255, 0, 0),     # class 0 -> blue
    (19, 69, 139),   # class 1 -> brown
    (255, 255, 0),   # class 2 -> cyan
    (255, 0, 255),   # class 3 -> magenta
    (0, 255, 0),     # class 4 -> green
    (255, 165, 0),   # class 5 -> orange
    (0, 255, 255),   # class 6 -> yellow
    (0, 0, 255),     # class 7 -> red
    (128, 0, 128),   # class 8 -> purple
]
# ----------------------------------------------------------------

# ------------------ Create model & load weights ------------------
img_size = 224
num_classes = len(class_names)
# recreate same architecture used for training
from timm.models.vision_transformer import vit_small_patch16_224
model = vit_small_patch16_224(num_classes=num_classes, img_size=img_size, pretrained=False)
# load weights (support saved state dict or checkpoint dict)
ck = torch.load(best_model_path, map_location=device)
if isinstance(ck, dict) and 'model_state' in ck:
    state = ck['model_state']
elif isinstance(ck, dict) and 'model_state_dict' in ck:
    state = ck['model_state_dict']
elif isinstance(ck, dict) and any(k.startswith('module.') for k in ck.keys()):
    # maybe a raw state dict with module prefix
    state = ck
else:
    state = ck

# If state dict keys have 'module.' prefix (DataParallel), strip it
new_state = {}
if isinstance(state, dict):
    for k, v in state.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v
    model.load_state_dict(new_state)
else:
    model.load_state_dict(state)

model = model.to(device)
model.eval()
print("Loaded model weights from:", best_model_path)

# ------------------ Preprocessing ------------------
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ------------------ Video IO setup ------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
print(f"Video opened: {video_path} | fps={fps:.2f} | size={width}x{height}")
print("Processing frames and writing to (no audio):", temp_video_path)

# ------------------ MediaPipe face detector ------------------
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

# Helper to clamp & int coords
def clamp_coords(x1, y1, x2, y2, w, h):
    x1i = max(0, min(w-1, int(round(x1))))
    y1i = max(0, min(h-1, int(round(y1))))
    x2i = max(0, min(w-1, int(round(x2))))
    y2i = max(0, min(h-1, int(round(y2))))
    return x1i, y1i, x2i, y2i

# Batch inference helper
def infer_batch(face_pils, batch_size=16):
    """face_pils: list of PIL RGB images"""
    if len(face_pils) == 0:
        return []
    tensors = [preprocess(im) for im in face_pils]
    x = torch.stack(tensors, dim=0).to(device)
    probs_all = []
    model.eval()
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
    probs_all = np.vstack(probs_all)  # (N, num_classes)
    return probs_all

# ------------------ Main loop ------------------
frame_idx = 0
t_start = time.time()

# Optional: you can maintain smoothing per track with a dict keyed by spatial bins
# but here we'll do per-frame detection+prediction
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h, w = frame.shape[:2]

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    face_pils = []
    boxes = []  # store integer boxes to draw later
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            # convert normalized bbox (xmin, ymin, w, h) to pixels
            x1 = bbox.xmin * w
            y1 = bbox.ymin * h
            x2 = x1 + bbox.width * w
            y2 = y1 + bbox.height * h

            # expand slightly for safety/context
            pad = 0.12
            bw = x2 - x1
            bh = y2 - y1
            x1 = x1 - pad * bw
            y1 = y1 - pad * bh
            x2 = x2 + pad * bw
            y2 = y2 + pad * bh

            x1i, y1i, x2i, y2i = clamp_coords(x1, y1, x2, y2, w, h)

            if x2i - x1i < 8 or y2i - y1i < 8:
                continue

            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            # convert to PIL RGB
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            face_pils.append(pil)
            boxes.append((x1i, y1i, x2i, y2i))

    # perform batched inference
    probs = infer_batch(face_pils, batch_size=16)  # shape (num_faces, num_classes)

    # draw boxes & labels
    for (x1i, y1i, x2i, y2i), prob_vec in zip(boxes, probs):
        class_idx = int(prob_vec.argmax())
        conf = float(prob_vec[class_idx])
        label = class_names[class_idx] if class_idx < len(class_names) else f"Class{class_idx}"
        color = colors[class_idx] if class_idx < len(colors) else (0,255,0)

        # Draw rectangle and label with confidence
        # Draw filled rectangle for text background
        text = f"{label}: {conf:.2f}"
        # text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # box
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
        # background rectangle for text
        text_bg_tl = (x1i, max(0, y1i - text_h - 10))
        text_bg_br = (x1i + text_w + 6, y1i)
        # semi-opaque background
        overlay = frame.copy()
        cv2.rectangle(overlay, text_bg_tl, text_bg_br, color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # put text in white or black depending on bg brightness
        cv2.putText(frame, text, (x1i + 3, y1i - 6), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # write processed frame to output (no-audio file)
    out.write(frame)

    # optional: print progress
    if frame_idx % 100 == 0:
        elapsed = time.time() - t_start
        print(f"Processed frame {frame_idx} - elapsed {elapsed:.1f}s")

# cleanup
cap.release()
out.release()
detector.close()
print("Finished processing frames (no audio). Merging audio...")

# ------------------ Merge original audio and save final video ------------------
orig_clip = VideoFileClip(video_path)
proc_clip = VideoFileClip(temp_video_path)
proc_clip = proc_clip.with_audio(orig_clip.audio)
proc_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', temp_audiofile="/tmp/temp-audio.m4a", remove_temp=True)

print("Done! Output with audio at:", output_video_path)

