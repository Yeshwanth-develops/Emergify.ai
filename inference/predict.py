# -------------------------
# Add project root FIRST
# -------------------------
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# -------------------------
# Imports
# -------------------------
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms, models
from severity.severity_model import SeverityNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 1. Load YOLO detector
# -------------------------
detector = YOLO("detection/emergency_yolo_fast/weights/best.pt")

# -------------------------
# 2. Load Scene Model
# -------------------------
scene_model = models.resnet18(weights=None)
scene_model.fc = torch.nn.Linear(scene_model.fc.in_features, 4)
scene_model.load_state_dict(
    torch.load("scene/models/scene_model.pth", map_location=device)
)
scene_model.eval().to(device)

scene_classes = ["accident", "fire", "flood", "normal"]

scene_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# 3. Load Severity Model
# -------------------------
sev_model = SeverityNet()
sev_model.load_state_dict(
    torch.load("severity/severity_model.pth", map_location=device)
)
sev_model.eval().to(device)

severity_labels = ["LOW", "MEDIUM", "HIGH"]

# -------------------------
# 4. Rule-based override
# -------------------------
def rule_based_override(features):
    people, vehicles, fire, smoke, sa, sf, sfl, sn = features

    if sfl == 1:  # Flood
        if people > 0 or vehicles > 0:
            return 2
        return 1

    if sn == 1 and people == 0 and fire == 0 and smoke == 0:
        return 0

    if sa == 1 and (people > 0 or vehicles > 0):
        return 2

    if sf == 1 and fire == 1:
        return 2

    if smoke == 1:
        return 2
    
    if fire == 1 and smoke == 1:
        return 2

    return None

# -------------------------
# 5. Inference Function
# -------------------------
def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ---- YOLO detection ----
    results = detector(img)[0]
    objs = {0: 0, 1: 0, 2: 0, 3: 0}

    for box in results.boxes:
        cls = int(box.cls)
        if cls in objs:
            objs[cls] += 1

    # ---- Object normalization (LEVEL-1) ----
    people = min(objs[0], 5) / 5.0
    vehicles = min(objs[1], 5) / 5.0
    fire = 1 if objs[2] > 0 else 0
    smoke = 1 if objs[3] > 0 else 0

    # ---- Scene prediction + confidence (LEVEL-1) ----
    input_tensor = scene_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        scene_logits = scene_model(input_tensor)
        scene_probs = F.softmax(scene_logits, dim=1)
        scene_id = scene_probs.argmax(dim=1).item()
        scene_conf = scene_probs[0][scene_id].item()

    print(
        f"Scene Prediction: {scene_classes[scene_id]} "
        f"(Confidence: {scene_conf:.2f})"
    )

    scene_flags = [0, 0, 0, 0]
    scene_flags[scene_id] = 1

    # ---- Feature vector ----
    features = [
        people,
        vehicles,
        fire,
        smoke,
        scene_flags[0],
        scene_flags[1],
        scene_flags[2],
        scene_flags[3]
    ]

    # ---- Severity decision + decision path (LEVEL-1) ----
    override = rule_based_override(features)

    if override is not None:
        severity = override
        confidence = 0.95
        decision_path = "Rule-based override"
    else:
        with torch.no_grad():
            preds = sev_model(
                torch.tensor([features], dtype=torch.float32).to(device)
            )
            probs = F.softmax(preds, dim=1)
            severity = probs.argmax(dim=1).item()
            confidence = probs[0][severity].item()
            decision_path = "ML-based severity model"

    # ---- Output ----
    print("Objects:", objs)
    print("Decision Path:", decision_path)
    print(
        "Final Severity:",
        severity_labels[severity],
        f"(Confidence: {confidence:.2f})"
    )

# -------------------------
# 6. Run
# -------------------------
if __name__ == "__main__":
    run("test4.png")
# -------------------------