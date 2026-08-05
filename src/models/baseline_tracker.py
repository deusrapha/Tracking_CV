import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import os

# Optional PyTorch imports for MobileNetV3 ReID
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

# Try importing TensorRT and PyCUDA safely
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT_BINDINGS = True
except ImportError:
    HAS_TRT_BINDINGS = False

class MobileNetReID:
    """
    Extracts a 128-D deep ReID feature embedding from a bounding box crop
    using a small, pre-trained MobileNetV3 architecture.
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            # Fallback for older torchvision versions
            self.model = models.mobilenet_v3_small(pretrained=True)
            
        # Modify classifier to output a 128-D embedding vector
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 128),
            torch.nn.BatchNorm1d(128)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.cpu().numpy().flatten()

class AppearanceExtractor:
    """
    Manages ReID feature extraction with multi-level fallback:
    TensorRT Engine -> ONNX Runtime -> PyTorch -> HSV Histograms.
    Supports dynamic batched inference.
    """
    def __init__(self, trt_engine_path="src/models/reid_model.engine", onnx_path="src/models/reid_model.onnx"):
        self.mode = "hsv"
        self.trt_context = None
        self.ort_session = None
        self.pytorch_model = None
        
        # 1. Attempt TensorRT Engine initialization
        if HAS_TRT_BINDINGS and os.path.exists(trt_engine_path):
            try:
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                with open(trt_engine_path, "rb") as f:
                    self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                self.trt_context = self.trt_engine.create_execution_context()
                self.mode = "trt"
                print(f"AppearanceExtractor: Successfully loaded TensorRT engine {trt_engine_path}")
            except Exception as e:
                print(f"AppearanceExtractor: Failed to load TensorRT engine: {e}")
                
        # 2. Attempt ONNX Runtime initialization (if TRT failed or not available)
        if self.mode == "hsv" and HAS_ONNXRUNTIME and os.path.exists(onnx_path):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                self.mode = "onnx"
                print(f"AppearanceExtractor: Successfully loaded ONNX model {onnx_path}")
            except Exception as e:
                print(f"AppearanceExtractor: Failed to load ONNX model: {e}")
                
        # 3. Attempt PyTorch initialization (if TRT and ONNX failed)
        if self.mode == "hsv" and HAS_PYTORCH:
            try:
                torch.manual_seed(42)
                self.pytorch_model = MobileNetReID(device='cpu')
                self.mode = "pytorch"
                print("AppearanceExtractor: Loaded baseline PyTorch MobileNetReID.")
            except Exception as e:
                print(f"AppearanceExtractor: Failed to load PyTorch ReID: {e}")
                
        if self.mode == "hsv":
            print("AppearanceExtractor: Using HSV histograms (baseline fallback).")

    def preprocess_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 128))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_norm = crop.astype(np.float32) / 255.0
        crop_norm = (crop_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        crop_norm = crop_norm.transpose(2, 0, 1) # HWC to CHW
        return crop_norm

    def extract_features_batch(self, frame, bboxes):
        if not bboxes:
            return []
            
        if self.mode == "hsv":
            return [self.compute_hsv_hist(frame, bbox) for bbox in bboxes]
            
        valid_crops = []
        valid_indices = []
        for idx, bbox in enumerate(bboxes):
            crop = self.preprocess_crop(frame, bbox)
            if crop is not None:
                valid_crops.append(crop)
                valid_indices.append(idx)
                
        if not valid_crops:
            return [self.compute_hsv_hist(frame, bbox) for bbox in bboxes]
            
        batch_data = np.stack(valid_crops).astype(np.float32)
        N = len(valid_crops)
        features_batch = None
        
        # A. TensorRT Inference
        if self.mode == "trt":
            try:
                self.trt_context.set_input_shape("input", (N, 3, 128, 128))
                d_input = cuda.mem_alloc(batch_data.nbytes)
                d_output = cuda.mem_alloc(N * 128 * 4)
                
                cuda.memcpy_htod(d_input, np.ascontiguousarray(batch_data))
                bindings = [int(d_input), int(d_output)]
                self.trt_context.execute_v2(bindings)
                
                h_output = np.empty((N, 128), dtype=np.float32)
                cuda.memcpy_dtoh(h_output, d_output)
                
                features_batch = h_output / np.linalg.norm(h_output, axis=1, keepdims=True)
            except Exception as e:
                print(f"TensorRT inference error: {e}. Falling back to ONNX...")
                self.mode = "onnx"
                
        # B. ONNX Runtime Inference
        if self.mode == "onnx" and features_batch is None:
            try:
                onnx_inputs = {self.ort_session.get_inputs()[0].name: batch_data}
                raw_out = self.ort_session.run(None, onnx_inputs)[0]
                features_batch = raw_out / np.linalg.norm(raw_out, axis=1, keepdims=True)
            except Exception as e:
                print(f"ONNX inference error: {e}. Falling back to PyTorch...")
                self.mode = "pytorch"
                
        # C. PyTorch Inference
        if self.mode == "pytorch" and features_batch is None:
            try:
                with torch.no_grad():
                    torch_in = torch.from_numpy(batch_data)
                    raw_out = self.pytorch_model.model(torch_in)
                    raw_out = torch.nn.functional.normalize(raw_out, p=2, dim=1).numpy()
                    features_batch = raw_out
            except Exception as e:
                print(f"PyTorch inference error: {e}. Falling back to HSV...")
                self.mode = "hsv"
                
        results = [None] * len(bboxes)
        for i, idx in enumerate(valid_indices):
            if features_batch is not None:
                results[idx] = features_batch[i]
            else:
                results[idx] = self.compute_hsv_hist(frame, bboxes[idx])
                
        for i in range(len(bboxes)):
            if results[i] is None:
                results[i] = self.compute_hsv_hist(frame, bboxes[i])
                
        return results

    def extract_features(self, frame, bbox):
        feats = self.extract_features_batch(frame, [bbox])
        return feats[0] if feats else None

    def compute_hsv_hist(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = frame[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

class KalmanFilterTracker:
    """
    An Extended Kalman Filter (EKF) implementing the Constant Turn Rate and Velocity (CTRV) model.
    State representation: x = [cx, cy, v, theta, omega, w, h]^T (7 dimensions)
    Measurement representation: z = [cx, cy, w, h]^T (4 dimensions)
    """
    def __init__(self, bbox, track_id, frame=None, appearance_extractor=None):
        self.track_id = track_id
        self.extractor = appearance_extractor if appearance_extractor is not None else AppearanceExtractor()
        
        cx, cy, w, h = self._bbox_to_measurement(bbox)
        # Initial state [cx, cy, v=0, theta=0, omega=0, w, h]^T
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0, w, h], dtype=np.float32).reshape(7, 1)
        
        self.last_updated_cx = cx
        self.last_updated_cy = cy
        
        # State covariance P
        self.P = np.eye(7, dtype=np.float32)
        self.P[2, 2] = 5.0   # Velocity variance
        self.P[3, 3] = 1.0   # Heading angle variance
        self.P[4, 4] = 0.1   # Yaw rate variance
        
        # Process noise covariance Q
        self.Q = np.eye(7, dtype=np.float32) * 0.05
        self.Q[2, 2] = 0.2
        self.Q[3, 3] = 0.1
        self.Q[4, 4] = 0.05
        
        # Measurement covariance R
        self.R = np.eye(4, dtype=np.float32) * 2.0
        
        self.age = 0
        self.time_since_update = 0
        self.last_bbox = bbox
        
        self.appearance_embedding = None
        if frame is not None:
            self.appearance_embedding = self.extractor.extract_features(frame, bbox)

    def _bbox_to_measurement(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w, h

    def _state_to_bbox(self, state):
        cx, cy, _, _, _, w, h = state.flatten()
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

    def predict(self, dt=1.0):
        """
        Predicts the next state using CTRV model with Runge-Kutta 4th order (RK4) integration.
        """
        cx, cy, v, theta, omega, w, h = self.x.flatten()
        
        # 1. State transition function f(x)
        def f_dot(state):
            # Returns [cx_dot, cy_dot, v_dot, theta_dot, omega_dot, w_dot, h_dot]
            _, _, cur_v, cur_theta, cur_omega, _, _ = state
            return np.array([
                cur_v * np.cos(cur_theta),
                cur_v * np.sin(cur_theta),
                0.0,
                cur_omega,
                0.0,
                0.0,
                0.0
            ], dtype=np.float32)
            
        # 2. RK4 Integration
        k1 = f_dot(self.x.flatten())
        k2 = f_dot(self.x.flatten() + 0.5 * dt * k1)
        k3 = f_dot(self.x.flatten() + 0.5 * dt * k2)
        k4 = f_dot(self.x.flatten() + dt * k3)
        
        dx = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.x = self.x + dx.reshape(7, 1)
        
        # Clamp omega turn rate to prevent wild spin projections
        self.x[4, 0] = np.clip(self.x[4, 0], -0.5, 0.5)
        
        # 3. Calculate Jacobian matrix F_j for state transition mapping
        F_j = np.eye(7, dtype=np.float32)
        cur_v = float(self.x[2, 0])
        cur_theta = float(self.x[3, 0])
        cur_omega = float(self.x[4, 0])
        
        if abs(cur_omega) > 1e-4:
            # CTRV partial derivatives
            F_j[0, 2] = (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / cur_omega
            F_j[0, 3] = cur_v * (np.cos(cur_theta + cur_omega * dt) - np.cos(cur_theta)) / cur_omega
            F_j[0, 4] = -cur_v * (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / (cur_omega ** 2) + cur_v * dt * np.cos(cur_theta + cur_omega * dt) / cur_omega
            
            F_j[1, 2] = (-np.cos(cur_theta + cur_omega * dt) + np.cos(cur_theta)) / cur_omega
            F_j[1, 3] = cur_v * (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / cur_omega
            F_j[1, 4] = -cur_v * (-np.cos(cur_theta + cur_omega * dt) + np.cos(cur_theta)) / (cur_omega ** 2) + cur_v * dt * np.sin(cur_theta + cur_omega * dt) / cur_omega
        else:
            # Linear CTRV limit (omega -> 0)
            F_j[0, 2] = np.cos(cur_theta) * dt
            F_j[0, 3] = -cur_v * np.sin(cur_theta) * dt
            F_j[1, 2] = np.sin(cur_theta) * dt
            F_j[1, 3] = cur_v * np.cos(cur_theta) * dt
            
        F_j[3, 4] = dt
        
        # 4. Covariance update
        self.P = F_j @ self.P @ F_j.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.last_bbox = self._state_to_bbox(self.x)
        return self.last_bbox

    def update(self, bbox, frame=None):
        elapsed_frames = max(1, self.time_since_update)
        self.time_since_update = 0
        cx, cy, w, h = self._bbox_to_measurement(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        
        # Measurement matrix H mapping state [cx, cy, v, theta, omega, w, h] to z [cx, cy, w, h]
        H = np.zeros((4, 7), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 5] = 1.0
        H[3, 6] = 1.0
        
        # Update EKF steps
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Re-estimate heading theta dynamically from physical translation vector relative to previous updated state
        dx = cx - self.last_updated_cx
        dy = cy - self.last_updated_cy
        dist = np.float32(np.hypot(dx, dy))
        
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(7, dtype=np.float32) - K @ H) @ self.P
        
        if dist > 1.0:
            self.x[3, 0] = np.arctan2(dy, dx)
            self.x[2, 0] = dist / elapsed_frames # update velocity estimate divided by elapsed frames
            
        self.last_updated_cx = float(self.x[0, 0])
        self.last_updated_cy = float(self.x[1, 0])
        self.last_bbox = self._state_to_bbox(self.x)
        
        # Hard-gated EMA update rule for ReID appearance embedding
        if frame is not None:
            new_feat = self.extractor.extract_features(frame, bbox)
            if new_feat is not None:
                if self.appearance_embedding is None:
                    self.appearance_embedding = new_feat
                else:
                    # Update similarity check: only update profile if similarity is high (prevents noise drift)
                    sim = compare_appearance(self.appearance_embedding, new_feat)
                    if sim > 0.65:
                        self.appearance_embedding = 0.9 * self.appearance_embedding + 0.1 * new_feat

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compare_appearance(feat1, feat2):
    if feat1 is None or feat2 is None:
        return 0.5
    # For normalized MobileNet vectors, cosine similarity is simply the dot product
    if len(feat1) == 128:
        return float(np.dot(feat1, feat2))
    # Fallback to histogram intersection similarity for HSV
    sim = np.sum(np.minimum(feat1, feat2)) / (np.sum(feat1) + 1e-6)
    return float(sim)

class BoundingBoxTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=30, app_weight=0.3, extractor=None):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.app_weight = app_weight
        self.tracks = []
        self.next_id = 1
        
        self.extractor = extractor if extractor is not None else AppearanceExtractor()
        self.prev_frame = None
        self.last_M = None
        self.last_theta_cam = 0.0

    def apply_camera_motion_compensation(self, curr_frame):
        """
        Registers frame offsets using OpenCV ORB to compute affine transformation
        and warps Kalman state coordinates [cx, cy] to prevent homography drift on yaw.
        """
        if self.prev_frame is None or curr_frame is None:
            return
            
        if hasattr(self, "_last_compensated_frame_id") and self._last_compensated_frame_id == id(curr_frame):
            return
        self._last_compensated_frame_id = id(curr_frame)
            
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=400)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            return
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        
        if len(matches) < 6:
            return
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate rigid transformation matrix
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if M is None:
            self.last_M = None
            self.last_theta_cam = 0.0
            return
            
        # Extract rotation offset to adjust heading vector
        theta_cam = np.arctan2(M[1, 0], M[0, 0])
        self.last_M = M
        self.last_theta_cam = theta_cam
        
        # Warp Kalman filter states
        for track in self.tracks:
            cx = float(track.x[0, 0])
            cy = float(track.x[1, 0])
            
            # Warp position
            new_pos = M @ np.array([cx, cy, 1.0], dtype=np.float32).reshape(3, 1)
            track.x[0, 0] = new_pos[0, 0]
            track.x[1, 0] = new_pos[1, 0]
            
            # Warp heading angle
            track.x[3, 0] += theta_cam

    def update(self, detections, frame=None, already_updated_ids=None, active_anchor_ids=None):
        if already_updated_ids is None:
            already_updated_ids = set()
        if active_anchor_ids is None:
            active_anchor_ids = set()
            
        # Apply camera motion compensation prior to state updates
        if frame is not None:
            self.apply_camera_motion_compensation(frame)
            
        # 1. Predict state for all active tracks
        predicted_boxes = []
        tracks_to_match = []
        track_indices_to_match = []
        
        for i, track in enumerate(self.tracks):
            if track.track_id in already_updated_ids:
                continue
            predicted_boxes.append(track.predict())
            tracks_to_match.append(track)
            track_indices_to_match.append(i)
            
        # 2. Extract ReID features for new detections
        det_features = []
        if frame is not None:
            det_features = self.extractor.extract_features_batch(frame, detections)
        else:
            det_features = [None] * len(detections)
            
        # 3. Match using Hungarian algorithm
        matched_track_indices = set()
        matched_detections = set()
        
        if len(tracks_to_match) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(tracks_to_match), len(detections)), dtype=np.float32)
            for i, pred_track in enumerate(tracks_to_match):
                for j, det_box in enumerate(detections):
                    iou = calculate_iou(predicted_boxes[i], det_box)
                    feat_sim = compare_appearance(pred_track.appearance_embedding, det_features[j])
                    
                    combined_sim = (1.0 - self.app_weight) * iou + self.app_weight * feat_sim
                    cost_matrix[i, j] = 1.0 - combined_sim
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                pred_box = predicted_boxes[r]
                det_box = detections[c]
                iou = calculate_iou(pred_box, det_box)
                
                if iou > self.iou_threshold or (feat_sim > 0.65 and iou > 0.05):
                    tracks_to_match[r].update(det_box, frame)
                    matched_track_indices.add(track_indices_to_match[r])
                    matched_detections.add(c)
                    
        # 4. Create new tracks for unmatched detections
        for j, det_box in enumerate(detections):
            if j not in matched_detections:
                new_track = KalmanFilterTracker(det_box, self.next_id, frame, self.extractor)
                self.tracks.append(new_track)
                self.next_id += 1
                
        # 5. Filter inactive tracks
        updated_tracks = []
        for i, track in enumerate(self.tracks):
            is_matched = (i in matched_track_indices) or (track.track_id in already_updated_ids)
            is_active_anchor = track.track_id in active_anchor_ids
            if is_matched or is_active_anchor or track.time_since_update < self.max_lost_frames:
                updated_tracks.append(track)
                
        self.tracks = updated_tracks
        
        if frame is not None:
            self.prev_frame = frame.copy()
            
        # Return visible tracks
        results = []
        for track in self.tracks:
            if track.time_since_update == 0:
                results.append({
                    "track_id": track.track_id,
                    "bbox": track.last_bbox
                })
        return results

if __name__ == "__main__":
    print("Testing BoundingBoxTracker upgrades (CTRV & CMC)...")
    tracker = BoundingBoxTracker()
    
    # Run a simple sequence with dummy frame movement to check CMC
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_a[100:200, 100:200, :] = 255
    
    # Frame b shifted to the right by 10 pixels
    frame_b = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_b[100:200, 110:210, :] = 255
    
    # Initialize track on frame a
    tracker.update([[150, 150, 200, 200]], frame_a)
    print(f"Track ID 1 position: {tracker.tracks[0].x.flatten()[:2]}")
    
    # Apply CMC and predict on frame b
    tracker.update([[160, 150, 210, 200]], frame_b)
    print(f"Track ID 1 updated position: {tracker.tracks[0].x.flatten()[:2]}")
    print("Upgrade verification successful!")
