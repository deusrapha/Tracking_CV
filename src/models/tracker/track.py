import numpy as np
import cv2
from .memory import TemporalMemory

def extract_texture_and_structural(frame, bbox):
    if frame is None or bbox is None:
        return None, None
        
    h, w, _ = frame.shape
    scale_x = w / 1920.0
    scale_y = h / 1080.0
    x1 = max(0, int(bbox[0] * scale_x))
    y1 = max(0, int(bbox[1] * scale_y))
    x2 = min(w, int(bbox[2] * scale_x))
    y2 = min(h, int(bbox[3] * scale_y))
    
    if x2 <= x1 or y2 <= y1:
        return None, None
        
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
        
    # 1. Texture: standard deviation of gradient magnitude (Sobel operators)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    texture_val = np.array([np.std(grad_mag)], dtype=np.float32)
    
    # 2. Structural body characteristics: upper vs lower color centroid contrast
    h_crop = gray.shape[0]
    top_half = crop[0 : h_crop // 2, :]
    bottom_half = crop[h_crop // 2 :, :]
    if top_half.size > 0 and bottom_half.size > 0:
        mean_top = np.mean(top_half, axis=(0, 1))
        mean_bottom = np.mean(bottom_half, axis=(0, 1))
        structural_val = (mean_top - mean_bottom).astype(np.float32)
    else:
        structural_val = np.zeros(3, dtype=np.float32)
        
    return texture_val, structural_val

def extract_color_histogram(frame, bbox):
    if frame is None or bbox is None:
        return None
        
    h, w, _ = frame.shape
    scale_x = w / 1920.0
    scale_y = h / 1080.0
    x1 = max(0, int(bbox[0] * scale_x))
    y1 = max(0, int(bbox[1] * scale_y))
    x2 = min(w, int(bbox[2] * scale_x))
    y2 = min(h, int(bbox[3] * scale_y))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

# --- Modular Identity Memory Structure ---
class AppearanceMemory:
    def __init__(self, embedding, color_hist, aspect_ratio, texture=None, structural=None):
        self.average_embedding = embedding.copy() if embedding is not None else None
        self.average_color_hist = color_hist.copy() if color_hist is not None else None
        self.average_body_ratio = aspect_ratio
        self.average_texture = texture.copy() if texture is not None else None
        self.average_structural = structural.copy() if structural is not None else None

    def update(self, embedding, color_hist, aspect_ratio, texture=None, structural=None):
        if embedding is not None:
            if self.average_embedding is None:
                self.average_embedding = embedding.copy()
            else:
                self.average_embedding = 0.95 * self.average_embedding + 0.05 * embedding
        if color_hist is not None:
            if self.average_color_hist is None:
                self.average_color_hist = color_hist.copy()
            else:
                self.average_color_hist = 0.95 * self.average_color_hist + 0.05 * color_hist
        if aspect_ratio is not None:
            self.average_body_ratio = 0.95 * self.average_body_ratio + 0.05 * aspect_ratio
        if texture is not None:
            if self.average_texture is None:
                self.average_texture = texture.copy()
            else:
                self.average_texture = 0.95 * self.average_texture + 0.05 * texture
        if structural is not None:
            if self.average_structural is None:
                self.average_structural = structural.copy()
            else:
                self.average_structural = 0.95 * self.average_structural + 0.05 * structural

    def compute_similarity(self, embedding, color_hist, aspect_ratio, texture, structural):
        s_emb = 1.0
        if self.average_embedding is not None and embedding is not None:
            s_emb = float(np.dot(self.average_embedding, embedding) / (np.linalg.norm(self.average_embedding) * np.linalg.norm(embedding) + 1e-6))
            s_emb = max(0.0, min(1.0, s_emb))
            
        s_col = 1.0
        if self.average_color_hist is not None and color_hist is not None:
            s_col = float(np.dot(self.average_color_hist, color_hist) / (np.linalg.norm(self.average_color_hist) * np.linalg.norm(color_hist) + 1e-6))
            s_col = max(0.0, min(1.0, s_col))
            
        s_tex = 1.0
        if self.average_texture is not None and texture is not None:
            diff = abs(self.average_texture[0] - texture[0])
            s_tex = 1.0 - min(1.0, diff / max(1e-3, self.average_texture[0]))
            
        s_shp = 1.0
        if self.average_body_ratio is not None and aspect_ratio is not None:
            diff = abs(self.average_body_ratio - aspect_ratio)
            s_shp = 1.0 - min(1.0, diff / 0.5)
            
        s_str = 1.0
        if self.average_structural is not None and structural is not None:
            s_str = float(np.dot(self.average_structural, structural) / (np.linalg.norm(self.average_structural) * np.linalg.norm(structural) + 1e-6))
            s_str = max(0.0, min(1.0, s_str))
            
        return 0.40 * s_emb + 0.25 * s_col + 0.15 * s_tex + 0.10 * s_shp + 0.10 * s_str

class MotionMemory:
    def __init__(self, speed, turn_rate):
        self.average_speed = speed
        self.average_acceleration = 0.0
        self.average_turn_rate = turn_rate
        self.prev_speed = speed

    def update(self, speed, turn_rate):
        acc = speed - self.prev_speed
        self.average_speed = 0.95 * self.average_speed + 0.05 * speed
        self.average_acceleration = 0.95 * self.average_acceleration + 0.05 * acc
        self.average_turn_rate = 0.95 * self.average_turn_rate + 0.05 * turn_rate
        self.prev_speed = speed

class BehaviouralDynamicsMemory:
    def __init__(self):
        self.average_grazing_speed = 1.0
        self.stop_frequency = 0.05
        self.turn_frequency = 0.05
        self.resting_ratio = 0.1
        self.movement_entropy = 0.5
        self.speed_variance = 0.0
        
        self.speed_history = []
        self.resting_duration = 0

    def update(self, speed, turn_rate):
        if speed < 0.2:
            self.resting_duration += 1
        else:
            self.resting_duration = 0
            
        is_turning = 1.0 if abs(turn_rate) > 0.05 else 0.0
        self.turn_frequency = 0.98 * self.turn_frequency + 0.02 * is_turning
        
        if speed >= 0.2:
            self.average_grazing_speed = 0.95 * self.average_grazing_speed + 0.05 * speed
        
        is_stopped = 1.0 if speed < 0.2 else 0.0
        self.stop_frequency = 0.98 * self.stop_frequency + 0.02 * is_stopped
        
        self.speed_history.append(speed)
        if len(self.speed_history) > 50:
            self.speed_history.pop(0)
            
        self.resting_ratio = sum(1.0 for s in self.speed_history if s < 0.2) / len(self.speed_history)
        self.speed_variance = float(np.var(self.speed_history))
        mean_speed = np.mean(self.speed_history)
        std_speed = np.std(self.speed_history)
        self.movement_entropy = float(std_speed / (mean_speed + 1e-6))

class SocialMemory:
    def __init__(self):
        self.nearest_neighbours = []
        self.leader = None
        self.follower = None
        self.herd_affiliation = 1.0

    def update(self, neighbours, leader, follower):
        self.nearest_neighbours = neighbours if neighbours is not None else []
        self.leader = leader
        self.follower = follower

class ReliabilityMemory:
    def __init__(self, initial_conf):
        self.successful_matches = 1
        self.successful_recoveries = 0
        self.false_recoveries = 0
        self.reliability_history = [initial_conf]
        self.average_reliability = initial_conf
        self.average_confidence = initial_conf
        self.last_recovery_confidence = 0.0
        self.identity_stability = 1.0

    def update(self, reliability, recovered=False, false_recovered=False):
        self.successful_matches += 1
        if recovered:
            self.successful_recoveries += 1
        if false_recovered:
            self.false_recoveries += 1
        self.reliability_history.append(reliability)
        if len(self.reliability_history) > 50:
            self.reliability_history.pop(0)
        self.average_reliability = sum(self.reliability_history) / len(self.reliability_history)
        self.average_confidence = self.average_reliability
        
        denom = self.successful_recoveries + self.successful_matches + self.false_recoveries
        self.identity_stability = float((self.successful_recoveries + self.successful_matches) / denom)

class IdentityMemory:
    def __init__(self, embedding, color_hist, aspect_ratio, speed, turn_rate, initial_conf, texture=None, structural=None):
        self.appearance = AppearanceMemory(embedding, color_hist, aspect_ratio, texture, structural)
        self.motion = MotionMemory(speed, turn_rate)
        self.behaviour = BehaviouralDynamicsMemory()
        self.social = SocialMemory()
        self.reliability = ReliabilityMemory(initial_conf)
        self.confidence = initial_conf

    def update(self, embedding, color_hist, aspect_ratio, speed, turn_rate, reliability, recovered=False, freeze=False, texture=None, structural=None, false_recovered=False):
        self.reliability.update(reliability, recovered, false_recovered)
        self.confidence = self.reliability.average_confidence
        if not freeze:
            self.appearance.update(embedding, color_hist, aspect_ratio, texture, structural)
            self.motion.update(speed, turn_rate)
            self.behaviour.update(speed, turn_rate)

    def decay_confidence(self, tsu, tau_m=150.0):
        self.confidence = float(self.reliability.average_confidence * np.exp(-tsu / tau_m))


class CounterfactualAmodalTrack:
    def __init__(self, bbox, track_id, frame=None, extractor=None, appearance_embedding=None, persistent_identity_id=None):
        self.track_id = track_id
        self.track_instance_id = track_id
        self.persistent_identity_id = persistent_identity_id if persistent_identity_id is not None else track_id
        self.extractor = extractor
        
        # Parse bbox [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        
        # 1. TrackState
        self.state = "NEW"  # NEW, VISIBLE, OCCLUDED, REMERGING, LOST, SEARCH, EXPIRED
        
        # 2. MotionState
        self.motion = {
            "x": np.array([cx, cy, 0.0, 0.0, 0.0, w, h], dtype=np.float32).reshape(7, 1),
            "P": np.eye(7, dtype=np.float32),
            "Q": np.eye(7, dtype=np.float32) * 0.05,
            "R": np.eye(4, dtype=np.float32) * 2.0,
            # Growing uncertainty for amodal gating
            "Sigma": np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32),
            "Q_sigma": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        }
        # Initialize motion cov components
        self.motion["P"][2, 2] = 5.0
        self.motion["P"][3, 3] = 1.0
        self.motion["P"][4, 4] = 0.1
        self.motion["Q"][2, 2] = 0.2
        self.motion["Q"][3, 3] = 0.1
        self.motion["Q"][4, 4] = 0.05
        
        self.last_updated_cx = cx
        self.last_updated_cy = cy
        self.last_bbox = bbox
        
        # 3. IdentityState
        self.identity = {
            "appearance_embedding": None,
            "confidence": 1.0,
            "age": 0,
            "last_matched_frame": 0,
            "verification_score": 1.0
        }
        if appearance_embedding is not None:
            self.identity["appearance_embedding"] = appearance_embedding
        elif frame is not None and extractor is not None:
            self.identity["appearance_embedding"] = extractor.extract_features(frame, bbox)
            
        # 4. SocialState
        self.social = {
            "blend_factor": 0.0
        }
        
        # 4b. BehaviorState
        self.behavior = {
            "mode": "walking",
            "heading_history": [],      # sliding window of last 8 headings
            "speed_history": [],        # sliding window of last 8 speeds
            "turn_persistence": 0.0,
            "is_leader": False
        }
        self.counterfactual_confidence = 1.0
        
        # 5. OcclusionState
        self.occlusion = {
            "frames_occluded": 0,
            "last_velocity": np.array([0.0, 0.0], dtype=np.float32),
            "entry_frame": 0
        }
        
        # 6. MemoryState
        self.memory = TemporalMemory()
        self.identity_hypothesis = None
        
        # Initialize identity_memory
        emb = self.identity["appearance_embedding"]
        texture, structural = extract_texture_and_structural(frame, bbox)
        color_hist = extract_color_histogram(frame, bbox)
        if color_hist is None:
            color_hist = emb
        self.identity_memory = IdentityMemory(emb, color_hist, w / h, 0.0, 0.0, 1.0, texture=texture, structural=structural)
        
    def get_identity_prototype(self):
        avg_emb = self.identity_memory.appearance.average_embedding
        if avg_emb is None:
            avg_emb = self.identity["appearance_embedding"]
            
        avg_col = self.identity_memory.appearance.average_color_hist
        if avg_col is None:
            avg_col = self.identity["appearance_embedding"]
            
        return avg_emb, avg_col
        
    def _state_to_bbox(self, x):
        cx, cy, _, _, _, w, h = x.flatten()
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

