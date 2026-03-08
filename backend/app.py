import os
import io
import base64
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

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

# Dynamic Objects we care about from COCO for YOLO (Mapping to readable names)
DYNAMIC_OBJECTS = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    15: "Bird",
    16: "Cat",
    17: "Dog",
    18: "Horse",
    19: "Sheep",
    20: "Cow",
    21: "Elephant",
    22: "Bear",
    23: "Zebra",
    24: "Giraffe"
}

# Global models
seg_model = None
seg_processor = None
yolo_model = None

def load_models():
    global seg_model, seg_processor, yolo_model
    
    print("Loading YOLOv8 (Dynamic Object Detection)...")
    yolo_model = YOLO('yolov8n.pt') 

    print("Loading SegFormer processor...")
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    
    print("Loading SegFormer architecture...")
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
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
        seg_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print("Warning: No custom trained weights found. Using base initialized model for UI testing.")
        
    seg_model.to(device)
    seg_model.eval()

@app.on_event("startup")
async def startup_event():
    load_models()

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
    
    # ---- 1. STATIC TERRAIN (SegFormer) ----
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
        
    logits = outputs.logits
    # Upsample to original image size
    upsampled_logits = F.interpolate(
        logits,
        size=original_size,
        mode="bilinear",
        align_corners=False
    )
    
    # ---- ACTIVE EDGE LEARNING: Confidence/Entropy Calculation ----
    probs = F.softmax(upsampled_logits, dim=1)
    # Entropy formula: -sum(p * log(p))
    entropy_map = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).squeeze().cpu().numpy()
    mean_entropy = float(entropy_map.mean())
    
    # If the model is highly uncertain across the image, save it for human review
    if mean_entropy > 0.8: 
        review_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'active_learning_review'))
        os.makedirs(review_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(review_dir, f"high_entropy_{timestamp}.jpg")
        
        # Save the original input image
        image.save(save_path, format="JPEG")
        print(f"[Active Learning] High entropy ({mean_entropy:.3f}) detected. Flagged frame saved to {save_path}")
    
    # Get highest probability class
    pred_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Colorize
    color_mask = colorize_mask(pred_mask)
    color_img = Image.fromarray(color_mask)
    
    # Create an overlaid version (50% blend)
    overlaid_img = Image.blend(image, color_img, alpha=0.5)
    
    # ---- 2. DYNAMIC HAZARDS (YOLOv8) ----
    # Run YOLO inference
    yolo_results = yolo_model(image)
    
    # Setup drawing context on the overlay image
    draw = ImageDraw.Draw(overlaid_img)
    dynamic_classes_found = set()
    
    # Process YOLO detections
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            
            # Check if it's an entity we care about
            if cls_id in DYNAMIC_OBJECTS:
                entity_name = DYNAMIC_OBJECTS[cls_id]
                dynamic_classes_found.add(entity_name)
                
                # Draw Box (Red for hazards)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw Label Background
                draw.rectangle([x1, y1 - 20, x1 + len(entity_name)*8, y1], fill="red")
                draw.text((x1 + 2, y1 - 18), entity_name, fill="white")
    
    # ---- ENCODE AND RETURN ----
    buffered_mask = io.BytesIO()
    color_img.save(buffered_mask, format="PNG")
    mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")
    
    buffered_overlay = io.BytesIO()
    overlaid_img.save(buffered_overlay, format="PNG")
    overlay_b64 = base64.b64encode(buffered_overlay.getvalue()).decode("utf-8")
    
    # Get unique classes present in the Segmentation prediction
    unique_classes = np.unique(pred_mask)
    detected_classes = [{"id": int(c), "name": CLASS_NAMES[int(c)], "color": f"rgb({COLOR_MAP[int(c)][0]}, {COLOR_MAP[int(c)][1]}, {COLOR_MAP[int(c)][2]})"} for c in unique_classes]
    
    # Append YOLO classes to the tags list with a special hazard color (Red)
    for idx, entity in enumerate(dynamic_classes_found):
        detected_classes.append({
            "id": 100 + idx, 
            "name": f"Hazard: {entity}", 
            "color": "rgb(255, 0, 0)"
        })
    
    return {
        "status": "success",
        "mask_base64": f"data:image/png;base64,{mask_b64}",
        "overlay_base64": f"data:image/png;base64,{overlay_b64}",
        "detected_classes": detected_classes
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected for Live Video Stream!")
    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            
            # Remove header if exists (e.g. data:image/jpeg;base64,)
            if "," in data:
                data = data.split(",")[1]
                
            image_data = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            original_size = image.size[::-1]
            
            # --- SAME LOGIC AS /predict ---
            inputs = seg_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = seg_model(**inputs)
                
            logits = outputs.logits
            upsampled_logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
            
            # Prediction mask
            pred_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
            color_mask = colorize_mask(pred_mask)
            color_img = Image.fromarray(color_mask)
            overlaid_img = Image.blend(image, color_img, alpha=0.5)
            
            # YOLO Detections
            yolo_results = yolo_model(image, verbose=False)
            draw = ImageDraw.Draw(overlaid_img)
            
            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in DYNAMIC_OBJECTS:
                        entity_name = DYNAMIC_OBJECTS[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        draw.rectangle([x1, y1 - 20, x1 + len(entity_name)*8, y1], fill="red")
                        draw.text((x1 + 2, y1 - 18), entity_name, fill="white")
                        
            buffered_overlay = io.BytesIO()
            overlaid_img.save(buffered_overlay, format="JPEG", quality=70) # Lower quality to keep stream fast
            overlay_b64 = base64.b64encode(buffered_overlay.getvalue()).decode("utf-8")
            
            await websocket.send_json({
                "status": "success",
                "overlay_base64": f"data:image/jpeg;base64,{overlay_b64}"
            })
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/health")
def health_check():
    return {"status": "healthy"}
