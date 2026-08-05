import os
import torch
import sys

# Ensure src path is available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.baseline_tracker import MobileNetReID

def export_reid_onnx(onnx_path="src/models/reid_model.onnx"):
    print("Initializing MobileNetV3 ReID model...")
    torch.manual_seed(42)
    # Instantiate ReID wrapper
    reid_wrapper = MobileNetReID(device='cpu')
    pytorch_model = reid_wrapper.model
    pytorch_model.eval()
    
    # Wrap in nn.Module to ensure clean dynamo tracing compatibility
    import torch.nn as nn
    class FullModelWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base(x)
            
    wrapped = FullModelWrapper(pytorch_model)
    wrapped.eval()
    
    # Create representative dummy input (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    
    print(f"Exporting model to ONNX format: {onnx_path} (opset_version=16)...")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    if os.path.exists(onnx_path):
        print(f"ONNX export completed successfully! File size: {os.path.getsize(onnx_path)/1024/1024:.2f} MB")
    else:
        print("ONNX export failed.")

if __name__ == "__main__":
    export_reid_onnx()
