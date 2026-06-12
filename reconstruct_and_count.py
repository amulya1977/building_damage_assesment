import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import segmentation_models_pytorch as smp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from shapely.geometry import Polygon, Point
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIG ---
BASE_DIR     = "../../Building_Damage_Project"
PROJECT_ROOT = os.path.join(BASE_DIR, "Final_Merge")
RAW_TRAIN    = os.path.join(BASE_DIR, "raw_data", "Train")
RAW_TEST     = os.path.join(BASE_DIR, "raw_data", "Test")

UNET_WEIGHTS   = os.path.join(PROJECT_ROOT, "best_building_model.pth")
MAXVIT_WEIGHTS = os.path.join(PROJECT_ROOT, "best_model_v3.pth")

TILE_SIZE = 512
STRIDE = 256  # 50% overlap for smooth blending
MIN_BUILDING_PIXELS = 10 # Lowered to 10 to match the training data prep threshold
CHIP_SIZE = 256
MAXVIT_IN_SIZE = 224

# Use CUDA if available, else MPS, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

DAMAGE_MAP = {
    'no damage':    0, 'no-damage':    0, 'no_damage':    0,
    'minor damage': 1, 'minor-damage': 1, 'minor_damage': 1,
    'major damage': 2, 'major-damage': 2, 'major_damage': 2,
    'destroyed':    3
}

# --- MODELS ---
class BuildingDamageModel(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_base',
            pretrained=False,
            num_classes=0,
            global_pool='avg',
            drop_path_rate=drop_path_rate
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        self.skip_proj = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats) + 0.1 * self.skip_proj(feats)


def load_unet(weights_path):
    print(f"Loading U-Net from {weights_path} onto {DEVICE}...")
    model = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    return model

def load_classifier(weights_path):
    print(f"Loading ConvNeXt from {weights_path} onto {DEVICE}...")
    model = BuildingDamageModel(num_classes=4)
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# --- GROUND TRUTH HELPER ---
def load_ground_truth_polygons(tif_path):
    """Loads polygons and their true labels from JSON files in the same dir."""
    folder = os.path.dirname(tif_path)
    filename = os.path.basename(tif_path)
    prefix = filename.replace('.tif', '').replace('_tif', '')
    
    base_json  = os.path.join(folder, f"{prefix}_json.json")
    align_json = os.path.join(folder, f"{prefix}_json_aligned.json")
    
    if not os.path.exists(base_json) or not os.path.exists(align_json):
        return []
        
    polys_with_labels = []
    try:
        with open(align_json, 'r') as f:
            align_data = json.load(f)
            
        s_x = np.mean([p[1][0] - p[0][0] for p in align_data])
        s_y = np.mean([p[1][1] - p[0][1] for p in align_data])
        
        with open(base_json, 'r') as f:
            entries = json.load(f)
            
        for entry in entries:
            raw_label = entry.get('label', 'no damage').lower().strip()
            label = DAMAGE_MAP.get(raw_label, -1)
            if label == -1:
                continue
                
            if 'pixels' in entry:
                pixel_coords = [
                    (p['x'] + s_x, p['y'] + s_y)
                    for p in entry['pixels']
                ]
                poly = Polygon(pixel_coords)
                if poly.is_valid and not poly.is_empty:
                    polys_with_labels.append({"poly": poly, "label": label})
                    
        return polys_with_labels
    except Exception as e:
        print(f"Warning: Could not load JSON for {prefix}: {e}")
        return []

# --- INFERENCE ---
def process_full_image(unet, classifier, tif_path, out_mask_path=None):
    print(f"\nProcessing {tif_path}...")
    
    # 1. Load Ground Truth
    gt_polygons = load_ground_truth_polygons(tif_path)
    print(f"Loaded {len(gt_polygons)} ground-truth polygons for evaluation.")

    # 2. MaxViT transform
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    val_transform = A.Compose([
        A.Resize(MAXVIT_IN_SIZE, MAXVIT_IN_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        is_uint16 = src.dtypes[0] == 'uint16'
        meta = src.meta.copy()

        # Arrays for Unet blending
        full_probs = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        y_coords = sorted(list(set(list(range(0, height - TILE_SIZE, STRIDE)) + [max(0, height - TILE_SIZE)])))
        x_coords = sorted(list(set(list(range(0, width - TILE_SIZE, STRIDE)) + [max(0, width - TILE_SIZE)])))

        total_tiles = len(y_coords) * len(x_coords)
        print(f"U-Net Tiling: {total_tiles} tiles to process.")

        hann_1d = np.hanning(TILE_SIZE)
        hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
        hann_2d = np.clip(hann_2d, 1e-4, 1.0)

        # FULL IMAGE IN MEMORY FOR FAST CHIP EXTRACTION LATER
        # (Assuming the TIFF can fit in RAM. Most 1024x1024 or 2048x2048 fit easily)
        print("Reading full image into memory for fast chip extraction...")
        full_img_rgb = src.read([1, 2, 3])
        if is_uint16:
            full_img_rgb = np.clip(full_img_rgb / 256.0, 0, 255).astype(np.uint8)
        else:
            full_img_rgb = np.clip(full_img_rgb, 0, 255).astype(np.uint8)
        full_img_hwc = full_img_rgb.transpose(1, 2, 0) # H, W, C

        # --- LOCALIZATION ---
        with torch.no_grad():
            with tqdm(total=total_tiles, desc="Localization (U-Net)") as pbar:
                for y in y_coords:
                    for x in x_coords:
                        img = full_img_rgb[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
                        c, h, w = img.shape
                        
                        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
                        img_tensor = (img_tensor - torch.tensor(MEAN).reshape(1,3,1,1)) / torch.tensor(STD).reshape(1,3,1,1)
                        img_tensor = img_tensor.to(DEVICE)

                        logits = unet(img_tensor)
                        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

                        valid_probs = probs[:h, :w]
                        valid_weights = hann_2d[:h, :w]

                        full_probs[y:y+h, x:x+w] += valid_probs * valid_weights
                        weight_map[y:y+h, x:x+w] += valid_weights

                        pbar.update(1)

        print("Averaging overlaps...")
        final_probs = full_probs / weight_map

        print("Applying Morphological Operations and Watershed...")
        binary_mask = (final_probs > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=1)
        sure_fg = cv2.erode(opening, kernel, iterations=2)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        num_labels, markers = cv2.connectedComponents(sure_fg, connectivity=4)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        valid_buildings = 0
        final_binary = np.zeros_like(binary_mask, dtype=np.uint8)
        
        y_true = []
        y_pred = []
        false_positives = 0

        # --- CLASSIFICATION & METRICS ---
        print("Extracting Chips and Classifying (ConvNeXt)...")
        with torch.no_grad():
            for i in range(2, num_labels + 1):
                building_mask = (markers == i).astype(np.uint8)
                area = cv2.countNonZero(building_mask)
                
                if area >= MIN_BUILDING_PIXELS:
                    valid_buildings += 1
                    final_binary[markers == i] = 1
                    
                    # 1. Find centroid of the building
                    M = cv2.moments(building_mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        continue # should not happen given area >= 10
                        
                    # 2. Match with Ground Truth
                    matched_label = None
                    centroid_pt = Point(cx, cy)
                    for gt in gt_polygons:
                        if gt["poly"].contains(centroid_pt):
                            matched_label = gt["label"]
                            break
                            
                    if matched_label is None:
                        false_positives += 1
                        continue # Skip adding it to classification metrics
                        
                    # 3. Extract Chip
                    pad = CHIP_SIZE // 2
                    x1 = max(cx - pad, 0)
                    y1 = max(cy - pad, 0)
                    x2 = min(cx + pad, width)
                    y2 = min(cy + pad, height)
                    
                    chip = full_img_hwc[y1:y2, x1:x2]
                    
                    # Pad the chip if it's too close to the image edge
                    if chip.shape[0] != CHIP_SIZE or chip.shape[1] != CHIP_SIZE:
                        # Create black pad
                        pad_chip = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
                        dh = chip.shape[0]
                        dw = chip.shape[1]
                        pad_chip[:dh, :dw] = chip
                        chip = pad_chip
                        
                    # 4. Transform and Predict
                    chip_tensor = val_transform(image=chip)['image'].unsqueeze(0).to(DEVICE)
                    logits = classifier(chip_tensor)
                    pred_label = logits.argmax(1).item()
                    
                    y_true.append(matched_label)
                    y_pred.append(pred_label)

        print(f"🏢 Buildings Detected: {valid_buildings}")
        print(f"⚠️  False Positives (No matching JSON polygon): {false_positives}")
        print(f"✅ Classified with Ground Truth: {len(y_true)}")

        # Save reconstructed mask if a path is provided
        if out_mask_path:
            meta.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": 'uint8',
                "compress": "lzw"
            })
            with rasterio.open(out_mask_path, "w", **meta) as dest:
                dest.write(final_binary, 1)

        return valid_buildings, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localize and Classify Building Damage")
    parser.add_argument("--input_dir", type=str, default=RAW_TEST, help="Path to folder containing TIFF images")
    parser.add_argument("--unet_weights", type=str, default=UNET_WEIGHTS, help="Path to U-Net weights")
    parser.add_argument("--classifier_weights", type=str, default=MAXVIT_WEIGHTS, help="Path to Classifier weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save the full binary masks (optional)")
    args = parser.parse_args()

    # Load both models
    unet = load_unet(args.unet_weights)
    classifier = load_classifier(args.classifier_weights)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    import glob
    tiff_files = glob.glob(os.path.join(args.input_dir, "*.tif*"))
    if not tiff_files:
        print(f"No TIFF files found in {args.input_dir}")
        exit(0)

    print(f"\n📂 Found {len(tiff_files)} TIFF images in {args.input_dir}")
    print("==========================================================")

    total_buildings = 0
    all_y_true = []
    all_y_pred = []
    results = []

    for tif_path in tiff_files:
        filename = os.path.basename(tif_path)
        out_mask_path = os.path.join(args.output_dir, f"mask_{filename}") if args.output_dir else None

        buildings_count, y_t, y_p = process_full_image(unet, classifier, tif_path, out_mask_path)
        
        total_buildings += buildings_count
        all_y_true.extend(y_t)
        all_y_pred.extend(y_p)
        results.append((filename, buildings_count, len(y_t)))

    print("\n======================================")
    print("📁 FINAL FOLDER SUMMARY")
    print("======================================")
    for filename, count, matched in results:
        print(f" - {filename}: {count} buildings detected ({matched} evaluated against GT)")
    print(f"\n🏆 Total buildings detected across all images: {total_buildings}")
    print(f"📊 Total buildings with matching Ground Truth: {len(all_y_true)}")
    
    if len(all_y_true) > 0:
        acc = accuracy_score(all_y_true, all_y_pred)
        f1  = f1_score(all_y_true, all_y_pred, average='macro')
        print(f"\n📈 CLASSIFICATION METRICS:")
        print(f"   Accuracy : {acc * 100:.2f}%")
        print(f"   Macro F1 : {f1:.4f}")
    else:
        print("\n⚠️ No buildings matched with ground truth JSON files. Could not compute metrics.")
    print("======================================\n")
