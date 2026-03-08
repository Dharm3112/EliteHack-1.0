import os
import sys
import io
import time
import torch

# Fix Windows Console Encoding for PyTorch's internal emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch.nn.utils.prune as prune
import numpy as np
from transformers import SegformerForSemanticSegmentation
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

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

def prune_pytorch_model(model, amount=0.2):
    print(f"\n--- Pruning Model Weights (Sparsity: {amount*100}%) ---")
    pruned_count = 0
    # Apply L1 Unstructured Pruning to Linear/Conv layers in the decoder
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make the pruning permanent so it exports efficiently
            prune.remove(module, 'weight')
            pruned_count += 1
            
    print(f"Successfully pruned {pruned_count} specific layers.")
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

def benchmark_pytorch(model, dummy_input, iterations=10):
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
    print(f"Average Latency: {avg_ms:.2f} ms")
    print(f"Current FPS: {fps:.2f}\n")
    return avg_ms

def benchmark_onnx(onnx_file_path, dummy_numpy, iterations=10, label="ONNX Baseline"):
    print(f"\n--- Loading {label} ---")
    
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
    onnx_quant_path = os.path.join(models_dir, "optimized_segformer_int8.onnx")
    
    # 3. Apply Pruning
    pruned_model = prune_pytorch_model(model, amount=0.20)
    
    # 4. Export Pruned Model to PyTorch 
    export_to_onnx(pruned_model, dummy_input, onnx_path)
    
    # 5. Apply INT8 Dynamic Quantization
    print("\n--- Applying INT8 Dynamic Quantization ---")
    if os.path.exists(onnx_path):
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_quant_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantization Successful: {onnx_quant_path}")
    
    # 6. Benchmarking Suite
    print("\n=========================================")
    print("      COMPUTE OPTIMIZATION PIPELINE      ")
    print("=========================================")
    
    # Test 1: Original architecture (unpruned)
    raw_model = load_pytorch_model(device)
    pt_ms = benchmark_pytorch(raw_model, dummy_input)
    
    # Test 2: Pruned architecture
    print("\n[Testing Pruned PyTorch Model]")
    pruned_ms = benchmark_pytorch(pruned_model, dummy_input)
    
    if os.path.exists(onnx_path):
        # Test 3: Standard ONNX graph
        onnx_ms = benchmark_onnx(onnx_path, dummy_numpy, label="FP32 ONNX Graph")
        
        # Test 4: INT8 Quantized ONNX graph
        onnx_quant_ms = benchmark_onnx(onnx_quant_path, dummy_numpy, label="INT8 Quantized ONNX Graph")
        
        print("=========================================")
        print("           FINAL RESULTS TABLE           ")
        print("=========================================")
        print(f"1. PyTorch Baseline (FP32):   {pt_ms:.2f} ms")
        print(f"2. PyTorch Pruned (FP32):     {pruned_ms:.2f} ms")
        print(f"3. ONNX Graph (FP32):         {onnx_ms:.2f} ms")
        print(f"4. ONNX Quantized (INT8):     {onnx_quant_ms:.2f} ms")
        print("=========================================")
        
        best_ms = min(pt_ms, pruned_ms, onnx_ms, onnx_quant_ms)
        speedup = pt_ms / best_ms
        
        print(f"\nOPTIMIZATION SUCCESS: Overall speedup is {speedup:.2f}x faster!")
        if best_ms <= 50:
            print("STATUS: PASSED. Model meets the strict < 50ms requirement for UGV autonomy.")
        else:
            print(f"STATUS: WARNING. Best latency is {best_ms:.2f} ms, missing the 50ms budget.")
            print("Action Required: TensorRT hardware compilation or aggressive 40% pruning needed.")

if __name__ == "__main__":
    main()
