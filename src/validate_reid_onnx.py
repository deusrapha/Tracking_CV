import os
import sys
import torch
import numpy as np
import onnxruntime as ort

# Ensure src path is available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baseline_tracker import MobileNetReID

def validate_reid_onnx(onnx_path="src/models/reid_model.onnx"):
    print("=== ONNX VALIDATION ===")
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return
        
    # 1. Initialize PyTorch model
    torch.manual_seed(42)
    reid_wrapper = MobileNetReID(device='cpu')
    pytorch_model = reid_wrapper.model
    pytorch_model.eval()
    
    # 2. Initialize ONNX runtime session
    print("Loading ONNX model in ONNX Runtime...")
    session = ort.InferenceSession(onnx_path)
    
    # 3. Create dummy inputs (Batch sizes 1, 4, 8)
    np.random.seed(42)
    for batch_size in [1, 4, 8]:
        print(f"\nTesting Batch Size: {batch_size}")
        dummy_in_np = np.random.randn(batch_size, 3, 128, 128).astype(np.float32)
        dummy_in_torch = torch.from_numpy(dummy_in_np)
        
        # PyTorch inference
        with torch.no_grad():
            py_out = pytorch_model(dummy_in_torch)
            py_out = torch.nn.functional.normalize(py_out, p=2, dim=1).numpy()
            
        # ONNX inference
        onnx_inputs = {session.get_inputs()[0].name: dummy_in_np}
        onnx_out = session.run(None, onnx_inputs)[0]
        # Normalize ONNX output as done in baseline tracker
        onnx_out_norm = onnx_out / np.linalg.norm(onnx_out, axis=1, keepdims=True)
        
        # Compare shapes
        print(f"  PyTorch Shape: {py_out.shape}, ONNX Shape: {onnx_out_norm.shape}")
        if py_out.shape != onnx_out_norm.shape:
            print("  [FAIL] Shapes mismatch!")
            continue
            
        # Parity check
        max_diff = np.max(np.abs(py_out - onnx_out_norm))
        print(f"  Max absolute difference: {max_diff:.6e}")
        
        # Cosine similarity
        cos_sims = []
        for i in range(batch_size):
            sim = np.dot(py_out[i], onnx_out_norm[i])
            cos_sims.append(sim)
        mean_sim = np.mean(cos_sims)
        print(f"  Mean Cosine Similarity: {mean_sim:.6f}")
        
        has_nans = np.isnan(onnx_out_norm).any()
        print(f"  Has NaNs/Infs: {has_nans}")
        
        if max_diff < 1e-4 and mean_sim > 0.999 and not has_nans:
            print("  [PASS] Numerical parity confirmed!")
        else:
            print("  [FAIL] Numerical parity check failed!")

if __name__ == "__main__":
    validate_reid_onnx()
