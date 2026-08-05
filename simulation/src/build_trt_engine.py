import os
import sys
import glob
import json
import cv2
import numpy as np

# Try importing TensorRT and PyCUDA/CUDA bindings safely
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

class CTRVReIDCalibrator(object):
    """
    Dummy/safe class wrapper for TensorRT calibrator in case PyCUDA/TensorRT is not present locally.
    It will be instantiated dynamically on the Jetson target.
    """
    pass

if HAS_TRT:
    class CTRVReIDCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calibration_images, batch_size, input_shape=(3, 128, 128), cache_file="src/models/reid_int8.cache"):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.calibration_images = calibration_images
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.cache_file = cache_file
            self.current_index = 0
            
            # Allocate device memory for calibration batches
            self.input_size = batch_size * input_shape[0] * input_shape[1] * input_shape[2] * 4 # float32
            self.d_input = cuda.mem_alloc(self.input_size)
            
        def get_batch_size(self):
            return self.batch_size
            
        def get_batch(self, names):
            if self.current_index + self.batch_size > len(self.calibration_images):
                return None
                
            batch_data = []
            for i in range(self.batch_size):
                img = self.calibration_images[self.current_index + i]
                # Preprocess image to CHW float32 normalized (similar to MobileNetReID transform)
                img = cv2.resize(img, (self.input_shape[1], self.input_shape[2]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                img = img.transpose(2, 0, 1) # HWC to CHW
                batch_data.append(img)
                
            self.current_index += self.batch_size
            
            # Flatten and copy to device memory
            flat_batch = np.ascontiguousarray(np.stack(batch_data), dtype=np.float32)
            cuda.memcpy_htod(self.d_input, flat_batch)
            return [int(self.d_input)]
            
        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
            
        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

def harvest_calibration_crops(dataset_dir="Dataset/processed_frames", max_crops=100):
    """
    Harvests representative crop images of cows from the dataset videos 
    to use as a calibration dataset for INT8 quantization.
    """
    crops = []
    video_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
    for v_dir in video_dirs:
        if not os.path.isdir(v_dir):
            continue
        tracks_json = os.path.join(v_dir, "predicted_tracks.json")
        if not os.path.exists(tracks_json):
            continue
            
        with open(tracks_json, "r", encoding="utf-8") as f:
            pred_tracks = json.load(f)
            
        frame_paths = sorted(glob.glob(os.path.join(v_dir, "frame_*.jpg")))
        for f_path in frame_paths:
            f_name = os.path.basename(f_path)
            tracks_in_frame = pred_tracks.get(f_name, [])
            if not tracks_in_frame:
                continue
                
            frame = cv2.imread(f_path)
            if frame is None:
                continue
                
            for track in tracks_in_frame:
                if track["status"] != "visible":
                    continue
                bbox = track["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 > 10 and y2 - y1 > 10:
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop)
                    if len(crops) >= max_crops:
                        return crops
    return crops

def build_tensorrt_engine(onnx_path="src/models/reid_model.onnx", 
                          engine_path="src/models/reid_model.engine", 
                          use_int8=False):
    """
    Compiles the ONNX graph into a highly optimized TensorRT engine.
    Supports FP16 and INT8 PTQ modes.
    """
    if not HAS_TRT:
        print("TensorRT or PyCUDA Python bindings are not installed on this machine.")
        print("Please run this script directly on your target Jetson board to build the engine.")
        return False
        
    print(f"Loading ONNX Model from: {onnx_path}...")
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 1. Initialize builder, config, and network definition
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Define network with explicit batch dimension
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network_value(flag)
    
    # 2. Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    # 3. Configure optimization profiles for dynamic batch size [1, 4, 16]
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name, (1, 3, 128, 128), (4, 3, 128, 128), (16, 3, 128, 128))
    config.add_optimization_profile(profile)
    
    # 4. Set precision modes
    print("Setting FP16 precision...")
    config.set_flag(trt.BuilderFlag.FP16)
    
    if use_int8:
        print("Configuring INT8 Quantization...")
        if not builder.platform_has_fast_int8():
            print("WARNING: This GPU does not support fast INT8. Falling back to FP16.")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Harvesting calibration crops from local dataset...")
            calib_crops = harvest_calibration_crops(max_crops=128)
            if len(calib_crops) < 16:
                print("WARNING: Not enough calibration crops found. Fallback to FP16.")
            else:
                print(f"Loaded {len(calib_crops)} representative crops. Running Calibrator...")
                calibrator = CTRVReIDCalibrator(calib_crops, batch_size=4)
                config.int8_calibrator = calibrator
                
    # 5. Build and save the serialized engine
    print(f"Building serialized engine (Output: {engine_path})...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("ERROR: Failed to build TensorRT engine.")
        return False
        
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"TensorRT Engine built successfully! File size: {os.path.getsize(engine_path)/1024/1024:.2f} MB")
    return True

if __name__ == "__main__":
    # Check arguments
    use_int8_flag = "--int8" in sys.argv
    build_tensorrt_engine(use_int8=use_int8_flag)
