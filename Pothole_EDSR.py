# full_pothole_pipeline.py (paste into a Jupyter cell)
import os
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

# -------------------------
# USER CONFIG - CHANGE THESE
# -------------------------
DATA_ROOT = r"C:\Pothole detection.v2-potholes-detection.coco"

TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_ANN_FILE   = os.path.join(DATA_ROOT, "train", "_annotations.coco.json")

TEST_IMAGES_DIR  = os.path.join(DATA_ROOT, "test", "images")
TEST_ANN_FILE    = os.path.join(DATA_ROOT, "test", "_annotations.coco.json")  # optional if test annotations exist

MODEL_SAVE_PATH = "pothole_frcnn.pth"

# Super-resolution model
EDSR_PATH = r"C:\EDSR_x4.pb"

# Camera params for diameter conversion
sensor_width_mm = 5.2
focal_length_mm = 4.2
camera_height_m = 1.5

# MiDaS scaling for meters (optional)
midas_scale_to_meters = None

# Training params
NUM_CLASSES = 2  # background + pothole
BATCH_SIZE = 2
NUM_EPOCHS = 8
LEARNING_RATE = 0.005

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Utility: Download EDSR model if missing
# -------------------------
def download_edser_if_missing(path=EDSR_PATH):
    url = "https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/samples/EDSR_x4.pb"
    path = Path(path)
    if path.exists():
        print("EDSR model found:", path)
        return str(path)
    print("Downloading EDSR model (~25 MB)...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Downloaded EDSR to:", path)
    return str(path)

# -------------------------
# Dataset class
# -------------------------i
class COCODataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.getImgIds()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(1)
            areas.append(ann.get('area', w*h))
            iscrowd.append(ann.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id]),
                  "area": areas, "iscrowd": iscrowd}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------------
# Build Faster R-CNN model
# -------------------------
def get_faster_rcnn_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# -------------------------
# Train model
# -------------------------
def train_model(train_images_dir, train_ann_file, save_path=MODEL_SAVE_PATH):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = COCODataset(train_images_dir, train_ann_file, transforms=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = get_faster_rcnn_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})
        lr_scheduler.step()
        print(f"Epoch {epoch+1} finished. Avg loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved:", save_path)
    return save_path

# -------------------------
# Load model for inference
# -------------------------
def load_model_for_inference(weights_path):
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# -------------------------
# Super-resolution helper
# -------------------------
def make_sr(edsr_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(edsr_path)
    sr.setModel('edsr', 4)
    return sr

# -------------------------
# Compute Ground Sampling Distance
# -------------------------
def compute_gsd(sensor_width_mm, focal_length_mm, camera_height_m, image_width_px):
    return (sensor_width_mm * camera_height_m) / (focal_length_mm * image_width_px)

# -------------------------
# Inference on image
# -------------------------
def infer_image(image_path, model, sr=None, gsd_params=None, score_thresh=0.5, resize_back=True):
    orig_bgr = cv2.imread(image_path)
    orig_h, orig_w = orig_bgr.shape[:2]

    # Super-resolution
    if sr is not None:
        sr_out = sr.upsample(orig_bgr)
        if resize_back:
            sr_out = cv2.resize(sr_out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        proc_bgr = sr_out
    else:
        proc_bgr = orig_bgr

    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(proc_rgb).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    vis = proc_bgr.copy()
    results = []

    for (box, score) in zip(boxes, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        diameter_px = x2 - x1
        diameter_m = None
        if gsd_params is not None:
            sensor_mm, focal_mm, cam_h_m, img_w_px = gsd_params
            gsd = compute_gsd(sensor_mm, focal_mm, cam_h_m, img_w_px)
            diameter_m = diameter_px * gsd

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        label_text = f"D_px={diameter_px}"
        if diameter_m is not None:
            label_text += f" | D_m={diameter_m:.2f}"
        cv2.putText(vis, label_text, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

        results.append({"box": (x1,y1,x2,y2), "score": float(score), "diameter_px": diameter_px, "diameter_m": diameter_m})

    return vis, results

def show_bgr(img_bgr, figsize=(12,8), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------
# MAIN PIPELINE
# -------------------------
if __name__ == "__main__":
    # 1️⃣ Train model if not exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Training Faster R-CNN on your dataset...")
        train_model(TRAIN_IMAGES_DIR, TRAIN_ANN_FILE, save_path=MODEL_SAVE_PATH)
    else:
        print("Loading existing model...")
    model = load_model_for_inference(MODEL_SAVE_PATH)

    # 2️⃣ Prepare super-resolution
    edsr_file = download_edser_if_missing()
    sr = make_sr(edsr_file) if edsr_file else None

    # 3️⃣ Inference on some test images
    sample_images = [os.path.join(TEST_IMAGES_DIR, f) for f in sorted(os.listdir(TEST_IMAGES_DIR))[:5]]

    for img_path in sample_images:
        print("Processing:", img_path)
        w = cv2.imread(img_path).shape[1]
        gsd_for_img = (sensor_width_mm, focal_length_mm, camera_height_m, w)
        vis, results = infer_image(img_path, model, sr=sr, gsd_params=gsd_for_img)
        show_bgr(vis, title=f"Detections: {len(results)}")
        print(json.dumps(results, indent=2))
