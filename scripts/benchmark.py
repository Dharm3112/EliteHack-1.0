import os
import sys
import time
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation
import onnxruntime as ort

# Add parent directory to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
NUM_CLASSES = 10
# Standardize input for the ONNX graph
BATCH_SIZE = 1
CHANNELS = 3
HEIGHT = 384
WIDTH = 640

def load_pytorch_model(device):
    print("Loading SegFormer architecture...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    best_model_path = os.path.join(base_dir, 'models', 'segformer_b0', 'best_segformer.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        print("Warning: custom weights not found. Benchmarking raw architecture.")
        
    model.to(device)
    model.eval()
    return model

def export_to_onnx(model, dummy_input, onnx_file_path):
    print(f"\nExporting PyTorch model to ONNX -> {onnx_file_path}")
    
    # Force UTF-8 encoding for inner torch.onnx loggers on Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # We use export directly. SegFormer inputs are dictionaries but we can pass pixel_values tensor directly
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=18,  # SegFormer requires opset >= 18 for Resize ops
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )
        print("ONNX Export Successful!")
        return True
    except Exception as e:
        print(f"ONNX Export Failed: {e}")
        return False

def benchmark_pytorch(model, dummy_input, iterations=100):
    print("\n--- Warming up PyTorch ---")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    print(f"--- Benchmarking PyTorch ({iterations} iterations) ---")
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000) # milliseconds
        
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"PyTorch Average Latency: {avg_ms:.2f} ms")
    print(f"PyTorch Current FPS: {fps:.2f}\n")
    return avg_ms

def benchmark_onnx(onnx_file_path, dummy_numpy, iterations=100):
    print(f"--- Loading ONNX Runtime ---")
    
    # Try GPU/TensorRT if available
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648, # 2GB
            'trt_fp16_enable': True,
        }),
        'CUDAExecutionProvider', 
        'CPUExecutionProvider'
    ] if torch.cuda.is_available() else ['CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(onnx_file_path, providers=providers)
        print(f"Loaded with providers: {ort_session.get_providers()}")
    except Exception as e:
        print(f"Failed to load ONNX session: {e}")
        return
        
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_numpy}
    
    print("--- Warming up ONNX ---")
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
        
    print(f"--- Benchmarking ONNX ({iterations} iterations) ---")
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"ONNX Average Latency: {avg_ms:.2f} ms")
    print(f"ONNX Current FPS: {fps:.2f}\n")
    return avg_ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # 1. Setup Model & Inputs
    model = load_pytorch_model(device)
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device=device)
    dummy_numpy = dummy_input.cpu().numpy()
    
    # 2. Paths
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = os.path.join(models_dir, "optimized_segformer.onnx")
    
    # 3. Export
    export_to_onnx(model, dummy_input, onnx_path)
    
    # 4. Benchmarking
    print("=========================================")
    print("         PERFORMANCE COMPARISON          ")
    print("=========================================")
    pt_ms = benchmark_pytorch(model, dummy_input)
    
    if os.path.exists(onnx_path):
        onnx_ms = benchmark_onnx(onnx_path, dummy_numpy)
        
        if onnx_ms:
            speedup = pt_ms / onnx_ms
            diff = pt_ms - onnx_ms
            print("=========================================")
            print("                RESULTS                  ")
            print("=========================================")
            print(f"ONNX is {speedup:.2f}x faster!")
            print(f"Latency reduced by {diff:.2f} ms per frame.")
            if onnx_ms <= 50:
                print("STATUS: SUCCESS. ONNX Model meets the < 50ms requirement for UGVs.")
            else:
                print("STATUS: WARNING. ONNX Model exceeds the 50ms budget. Needs TensorRT quantization.")
            print("=========================================")

if __name__ == "__main__":
    main()
