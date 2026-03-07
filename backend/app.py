import os
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

app = FastAPI(title="Offroad Segmentation API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

COLOR_MAP = {
    0: [0, 0, 0],         # Background - Black
    1: [0, 100, 0],       # Trees - Dark Green
    2: [144, 238, 144],   # Lush Bushes - Light Green
    3: [240, 230, 140],   # Dry Grass - Khaki
    4: [128, 128, 0],     # Dry Bushes - Olive
    5: [139, 69, 19],     # Ground Clutter - Brown
    6: [101, 67, 33],     # Logs - Dark Wood
    7: [128, 128, 128],   # Rocks - Gray
    8: [194, 178, 128],   # Landscape - Sand
    9: [135, 206, 235]    # Sky - Light Blue
}

# Global model and processor
model = None
processor = None

def load_model():
    global model, processor
    print("Loading processor...")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    
    print("Loading model architecture...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Try to load custom weights if training completed/saved
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    best_model_path = os.path.join(base_dir, 'models', 'segformer_b0', 'best_segformer.pth')
    last_model_path = os.path.join(base_dir, 'models', 'segformer_b0', 'last_segformer.pth')
    
    weights_path = None
    if os.path.exists(best_model_path):
        weights_path = best_model_path
    elif os.path.exists(last_model_path):
        weights_path = last_model_path
        
    if weights_path:
        print(f"Loading custom weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print("Warning: No custom trained weights found. Using base initialized model for UI testing.")
        
    model.to(device)
    model.eval()

@app.on_event("startup")
async def startup_event():
    load_model()

def colorize_mask(mask_numpy):
    h, w = mask_numpy.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in COLOR_MAP.items():
        color_mask[mask_numpy == cls_idx] = color
    return color_mask

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size[::-1] # (height, width)
    
    # Convert image to numpy for processor or directly use PIL
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Upsample to original image size
    upsampled_logits = F.interpolate(
        logits,
        size=original_size,
        mode="bilinear",
        align_corners=False
    )
    
    # Get highest probability class
    pred_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Colorize
    color_mask = colorize_mask(pred_mask)
    
    # Convert color mask to base64 image
    color_img = Image.fromarray(color_mask)
    
    # Create an overlaid version (50% blend)
    overlaid_img = Image.blend(image, color_img, alpha=0.5)
    
    buffered_mask = io.BytesIO()
    color_img.save(buffered_mask, format="PNG")
    mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")
    
    buffered_overlay = io.BytesIO()
    overlaid_img.save(buffered_overlay, format="PNG")
    overlay_b64 = base64.b64encode(buffered_overlay.getvalue()).decode("utf-8")
    
    # Get unique classes present in the prediction
    unique_classes = np.unique(pred_mask)
    detected_classes = [{"id": int(c), "name": CLASS_NAMES[int(c)], "color": f"rgb({COLOR_MAP[int(c)][0]}, {COLOR_MAP[int(c)][1]}, {COLOR_MAP[int(c)][2]})"} for c in unique_classes]
    
    return {
        "status": "success",
        "mask_base64": f"data:image/png;base64,{mask_b64}",
        "overlay_base64": f"data:image/png;base64,{overlay_b64}",
        "detected_classes": detected_classes
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
