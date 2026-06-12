import streamlit as st
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import rasterio
from rasterio.windows import Window
import segmentation_models_pytorch as smp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from PIL import Image
import torch.nn.functional as F

# --- SETTINGS ---
st.set_page_config(page_title="Building Damage Dashboard", layout="wide")

TILE_SIZE = 512
STRIDE = 256
MIN_BUILDING_PIXELS = 10
CHIP_SIZE = 256
MAXVIT_IN_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES = ['No Damage', 'Minor', 'Major', 'Destroyed']

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

@st.cache_resource
def load_models():
    # Resolve the correct path to the Final_Merge folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_merge_dir = os.path.join(current_dir, "..", "Final_Merge")
    unet_path = os.path.join(final_merge_dir, "best_building_model.pth")
    classifier_path = os.path.join(final_merge_dir, "best_model_v3.pth")
    
    # U-Net
    unet = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    ckpt_u = torch.load(unet_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt_u, dict) and "model_state" in ckpt_u:
        unet.load_state_dict(ckpt_u["model_state"])
    else:
        unet.load_state_dict(ckpt_u)
    unet.to(DEVICE)
    unet.eval()

    # Classifier
    classifier = BuildingDamageModel(num_classes=4)
    ckpt_m = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt_m, dict) and 'state_dict' in ckpt_m:
        classifier.load_state_dict(ckpt_m['state_dict'])
    elif isinstance(ckpt_m, dict) and 'model_state_dict' in ckpt_m:
        classifier.load_state_dict(ckpt_m['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt_m)
    classifier.to(DEVICE)
    classifier.eval()
    
    return unet, classifier

# --- INFERENCE PIPELINE ---
def process_uploaded_image(uploaded_file, unet, classifier, progress_bar, status_text):
    # Save uploaded file temporarily
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_upload.tif")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    val_transform = A.Compose([
        A.Resize(MAXVIT_IN_SIZE, MAXVIT_IN_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    with rasterio.open(temp_path) as src:
        width = src.width
        height = src.height
        is_uint16 = src.dtypes[0] == 'uint16'

        full_probs = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        y_coords = sorted(list(set(list(range(0, height - TILE_SIZE, STRIDE)) + [max(0, height - TILE_SIZE)])))
        x_coords = sorted(list(set(list(range(0, width - TILE_SIZE, STRIDE)) + [max(0, width - TILE_SIZE)])))

        total_tiles = len(y_coords) * len(x_coords)
        
        hann_1d = np.hanning(TILE_SIZE)
        hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
        hann_2d = np.clip(hann_2d, 1e-4, 1.0)

        status_text.text("Reading image into memory...")
        full_img_rgb = src.read([1, 2, 3])
        if is_uint16:
            full_img_rgb = np.clip(full_img_rgb / 256.0, 0, 255).astype(np.uint8)
        else:
            full_img_rgb = np.clip(full_img_rgb, 0, 255).astype(np.uint8)
        full_img_hwc = full_img_rgb.transpose(1, 2, 0)

        # LOCALIZATION
        status_text.text("Detecting buildings (U-Net)...")
        tiles_processed = 0
        with torch.no_grad():
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

                    tiles_processed += 1
                    progress_bar.progress(int((tiles_processed / total_tiles) * 50)) # first 50%

        status_text.text("Separating buildings (Watershed)...")
        final_probs = full_probs / weight_map
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
        
        # CLASSIFICATION
        status_text.text("Classifying damage (ConvNeXt)...")
        results = {
            "No Damage": 0, "Minor": 0, "Major": 0, "Destroyed": 0
        }
        total_confidence = 0.0
        valid_buildings = 0
        under_confident_chips = []

        total_buildings_expected = num_labels - 1
        processed_buildings = 0

        with torch.no_grad():
            for i in range(2, num_labels + 1):
                building_mask = (markers == i).astype(np.uint8)
                area = cv2.countNonZero(building_mask)
                
                if area >= MIN_BUILDING_PIXELS:
                    valid_buildings += 1
                    M = cv2.moments(building_mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        continue
                        
                    pad = CHIP_SIZE // 2
                    x1 = max(cx - pad, 0)
                    y1 = max(cy - pad, 0)
                    x2 = min(cx + pad, width)
                    y2 = min(cy + pad, height)
                    
                    chip = full_img_hwc[y1:y2, x1:x2]
                    
                    if chip.shape[0] != CHIP_SIZE or chip.shape[1] != CHIP_SIZE:
                        pad_chip = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
                        dh = chip.shape[0]
                        dw = chip.shape[1]
                        pad_chip[:dh, :dw] = chip
                        chip = pad_chip
                        
                    chip_tensor = val_transform(image=chip)['image'].unsqueeze(0).to(DEVICE)
                    logits = classifier(chip_tensor)
                    
                    # Apply Softmax for confidence
                    probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
                    pred_idx = probs.argmax()
                    confidence = float(probs[pred_idx])
                    pred_class = CLASS_NAMES[pred_idx]
                    
                    results[pred_class] += 1
                    total_confidence += confidence

                    # Flag under-confident damaged buildings (< 40%)
                    # We check if it is underconfident. We flag all underconfident damaged buildings.
                    if pred_class != "No Damage" and confidence < 0.40:
                        under_confident_chips.append({
                            "image": chip,
                            "pred": pred_class,
                            "conf": confidence
                        })

                processed_buildings += 1
                progress_bar.progress(50 + int((processed_buildings / max(total_buildings_expected, 1)) * 50)) # remaining 50%
                
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    avg_conf = total_confidence / max(valid_buildings, 1)
    return valid_buildings, results, avg_conf, under_confident_chips

# --- UI ---
st.title("🛰️ Building Damage Assessment Dashboard")
st.markdown("Upload a high-resolution satellite image (TIFF). The AI pipeline will automatically localize and classify damage levels for all buildings.")

try:
    unet, classifier = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure the 'best_building_model.pth' and 'best_model_v3.pth' exist in the Final_Merge directory.")
    st.stop()

uploaded_file = st.file_uploader("Upload a TIFF image", type=["tif", "tiff"])

if uploaded_file is not None:
    st.subheader(f"Analyzing `{uploaded_file.name}`...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run Inference
    total_bldgs, results, avg_conf, under_conf_chips = process_uploaded_image(uploaded_file, unet, classifier, progress_bar, status_text)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis Complete!")
    
    # Process Metrics
    damaged_count = results["Minor"] + results["Major"] + results["Destroyed"]
    perc_damaged = (damaged_count / total_bldgs) * 100 if total_bldgs > 0 else 0
    under_conf_count = len(under_conf_chips)
    
    if perc_damaged > 75:
        impact = "Severe"
    elif perc_damaged > 50:
        impact = "High"
    elif perc_damaged > 25:
        impact = "Moderate"
    else:
        impact = "Low"
        
    # Table Data
    df = pd.DataFrame([{
        "Area / TIF": uploaded_file.name,
        "Total Buildings": total_bldgs,
        "Damaged Buildings": damaged_count,
        "% Damaged": f"{perc_damaged:.1f}%",
        "Impact Level": impact,
        "Under-Conf Damaged": under_conf_count,
        "Avg Confidence": f"{avg_conf:.2f}",
        "No Damage Count": results["No Damage"],
        "Minor Count": results["Minor"],
        "Major Count": results["Major"],
        "Destroyed Count": results["Destroyed"]
    }])
    
    st.markdown("### 📊 Overview")
    
    # Use st.dataframe with custom styling for aesthetics
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display Under-confident chips
    if under_conf_count > 0:
        st.markdown(f"### ⚠️ Under-Confident Predictions (< 40%)")
        st.info(f"The model detected {under_conf_count} damaged buildings where its confidence was below 40%. Manual review of these specific structures is recommended.")
        
        # Display images in a grid (4 columns)
        cols = st.columns(4)
        for idx, item in enumerate(under_conf_chips):
            col = cols[idx % 4]
            img = Image.fromarray(item["image"])
            # Format the caption with the predicted class and confidence
            col.image(img, caption=f"{item['pred']} ({item['conf'] * 100:.1f}%)", use_container_width=True)
    else:
        st.success("No under-confident damaged predictions found. The model is highly confident in its assessments.")
