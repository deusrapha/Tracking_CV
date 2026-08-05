import numpy as np
try:
    from baseline_tracker import KalmanFilterTracker, compare_appearance
except ImportError:
    from models.baseline_tracker import KalmanFilterTracker, compare_appearance

class AmodalAnchor:
    """
    Maintains a virtual animal hypothesis inside an occlusion zone.
    Updates location using kinematic prediction and herd social priors.
    Models uncertainty as a growing Gaussian covariance.
    """
    def __init__(self, track_id, last_bbox, last_velocity, entry_frame, appearance_embedding=None):
        self.track_id = track_id
        self.last_bbox = last_bbox # [x_min, y_min, x_max, y_max]
        
        # Current estimated position on image plane (cx, cy)
        x1, y1, x2, y2 = last_bbox
        self.cx = x1 + (x2 - x1) / 2.0
        self.cy = y1 + (y2 - y1) / 2.0
        self.w = x2 - x1
        self.h = y2 - y1
        
        # Last known velocity on image plane
        self.velocity = np.array(last_velocity, dtype=np.float32) # [vx, vy]
        
        # Initial frame indices
        self.entry_frame = entry_frame
        self.frames_occluded = 0
        self.is_expired = False
        
        # Covariance matrix for spatial uncertainty (starts small, grows over time)
        self.Sigma = np.eye(2, dtype=np.float32) * 5.0 # initial variance
        
        # Process noise covariance (controls how fast uncertainty grows)
        self.Q = np.eye(2, dtype=np.float32) * 2.0
        
        # Stored appearance descriptor (HSV color histogram)
        self.appearance_embedding = appearance_embedding

    def predict(self, visible_herd_mean_velocity=None, herd_cohesion_weight=0.3, max_occlusion_frames=150):
        """
        Updates the anchor's estimated location and increments occlusion age.
        Implements an adaptive expiry threshold: double the timeout to 300 frames
        if the target is stationary/resting (velocity magnitude < 0.2 px/f).
        """
        self.frames_occluded += 1
        
        # Calculate velocity magnitude
        vel_mag = np.linalg.norm(self.velocity)
        timeout_limit = 300 if vel_mag < 0.2 else max_occlusion_frames
        
        # Expiry Check
        if self.frames_occluded > timeout_limit:
            self.is_expired = True
            
        # Blended velocity calculation
        if visible_herd_mean_velocity is not None:
            v_herd = np.array(visible_herd_mean_velocity, dtype=np.float32)
            blended_velocity = (1.0 - herd_cohesion_weight) * self.velocity + herd_cohesion_weight * v_herd
        else:
            blended_velocity = self.velocity
            
        # Update estimated position
        self.cx += blended_velocity[0]
        self.cy += blended_velocity[1]
        
        # Update uncertainty covariance (growing error search window)
        self.Sigma = self.Sigma + self.Q
        
        # Reconstruct estimated bbox
        x1 = self.cx - self.w / 2.0
        y1 = self.cy - self.h / 2.0
        x2 = self.cx + self.w / 2.0
        y2 = self.cy + self.h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

    def get_occupancy_distribution(self, grid_w, grid_h):
        """
        Generates a 2D Gaussian probability distribution (occupancy grid)
        centered at the anchor's estimated position (cx, cy) with covariance Sigma.
        """
        x = np.arange(0, grid_w, 1, dtype=np.float32)
        y = np.arange(0, grid_h, 1, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        
        mu = np.array([self.cx, self.cy], dtype=np.float32)
        
        try:
            inv_Sigma = np.linalg.inv(self.Sigma)
            det_Sigma = np.linalg.det(self.Sigma)
            norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_Sigma))
            
            dev = pos - mu
            exponent = -0.5 * np.sum(dev @ inv_Sigma * dev, axis=-1)
            pdf = norm_const * np.exp(exponent)
            
            total = np.sum(pdf)
            if total > 0:
                pdf = pdf / total
            return pdf
        except np.linalg.LinAlgError:
            return np.zeros((grid_h, grid_w), dtype=np.float32)

class OcclusionAwareTracker:
    def __init__(self, baseline_tracker, occlusion_projector, grid_w=1920, grid_h=1080):
        self.baseline_tracker = baseline_tracker
        self.projector = occlusion_projector
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        # Holds active amodal anchors: {track_id: AmodalAnchor}
        self.amodal_anchors = {}
        self.frame_count = 0

    def step(self, frame, detections, altitude, pitch, roll):
        """
        Processes a single video frame.
        """
        self.frame_count += 1
        
        # 1. Segment vegetation to identify occlusion zones (shadow projection mapping)
        occ_mask = self.projector.segment_vegetation(frame)
        
        # 2. Check for re-emergence of active amodal anchors against incoming detections
        remaining_detections = list(detections)
        reemerged_ids = []
        
        # Calculate mean velocity of visible herd in previous frame
        visible_velocities = []
        for track in self.baseline_tracker.tracks:
            if track.time_since_update == 0:
                v = float(track.x[2, 0])
                theta = float(track.x[3, 0])
                vx = v * np.cos(theta)
                vy = v * np.sin(theta)
                visible_velocities.append([vx, vy])
        mean_herd_velocity = np.mean(visible_velocities, axis=0) if len(visible_velocities) > 0 else np.array([0, 0], dtype=np.float32)
        
        # Compute appearance embedding for each incoming detection to enable ReID comparison
        det_appearances = []
        if len(remaining_detections) > 0:
            det_appearances = self.baseline_tracker.extractor.extract_features_batch(frame, remaining_detections)
            
        # Check active amodal anchors
        for track_id, anchor in list(self.amodal_anchors.items()):
            pred_bbox = anchor.predict(mean_herd_velocity, herd_cohesion_weight=0.3)
            
            # Clean up expired anchors
            if anchor.is_expired:
                print(f"  [ANCHOR EXPIRED] Track {track_id} expired after {anchor.frames_occluded} frames in occlusion.")
                del self.amodal_anchors[track_id]
                continue
                
            best_score = 0.0
            best_det_idx = -1
            best_iou = 0.0
            
            for idx, det in enumerate(remaining_detections):
                iou = self._calculate_iou(pred_bbox, det)
                app_sim = compare_appearance(anchor.appearance_embedding, det_appearances[idx])
                
                # Combined validation score: 70% motion/overlap, 30% visual appearance
                combined_score = 0.7 * iou + 0.3 * app_sim
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_det_idx = idx
                    best_iou = iou
                    
            if best_score > 0.4 and best_iou > 0.2: # Match threshold
                best_det = remaining_detections[best_det_idx]
                print(f"  [ANCHOR REEMERGE] Track {track_id} re-emerged from occlusion at frame {self.frame_count} with score={best_score:.2f} (IoU={best_iou:.2f}).")
                
                # Update baseline tracker corresponding track with the detection
                for track in self.baseline_tracker.tracks:
                    if track.track_id == track_id:
                        track.update(best_det, frame)
                        track.time_since_update = 0
                        break
                
                remaining_detections.pop(best_det_idx)
                det_appearances.pop(best_det_idx)
                reemerged_ids.append(track_id)
                del self.amodal_anchors[track_id]
        
        # 3. Update baseline tracker with remaining detections
        active_visible_tracks = self.baseline_tracker.update(remaining_detections, frame, already_updated_ids=set(reemerged_ids))
        active_ids = {t["track_id"] for t in active_visible_tracks}
        
        # 4. Check for occlusion entry: if a track has time_since_update == 1 (was lost this frame)
        # and its last position intersects with the occlusion mask, seed an anchor
        for track in self.baseline_tracker.tracks:
            track_id = track.track_id
            
            if track.time_since_update == 1 and track_id not in self.amodal_anchors and track_id not in reemerged_ids:
                cx, cy = int(track.last_bbox[0] + (track.last_bbox[2] - track.last_bbox[0])/2), int(track.last_bbox[1] + (track.last_bbox[3] - track.last_bbox[1])/2)
                cx = min(max(cx, 0), self.grid_w - 1)
                cy = min(max(cy, 0), self.grid_h - 1)
                
                if occ_mask[cy, cx] > 0:
                    print(f"  [ANCHOR SEED] Track {track_id} entered occlusion at frame {self.frame_count}. Seeding virtual anchor.")
                    v = float(track.x[2, 0])
                    theta = float(track.x[3, 0])
                    vx = v * np.cos(theta)
                    vy = v * np.sin(theta)
                    
                    anchor = AmodalAnchor(
                        track_id=track_id,
                        last_bbox=track.last_bbox,
                        last_velocity=[vx, vy],
                        entry_frame=self.frame_count,
                        appearance_embedding=track.appearance_embedding
                    )
                    self.amodal_anchors[track_id] = anchor
                    
        # 5. Duplicate Suppression (Merge overlapping anchors)
        # If two anchors get too close (IoU > 0.6), suppress the one with less history
        anchor_ids = list(self.amodal_anchors.keys())
        to_suppress = set()
        for i in range(len(anchor_ids)):
            for j in range(i + 1, len(anchor_ids)):
                id1 = anchor_ids[i]
                id2 = anchor_ids[j]
                if id1 in to_suppress or id2 in to_suppress:
                    continue
                a1 = self.amodal_anchors[id1]
                a2 = self.amodal_anchors[id2]
                
                box1 = [a1.cx - a1.w/2, a1.cy - a1.h/2, a1.cx + a1.w/2, a1.cy + a1.h/2]
                box2 = [a2.cx - a2.w/2, a2.cy - a2.h/2, a2.cx + a2.w/2, a2.cy + a2.h/2]
                
                if self._calculate_iou(box1, box2) > 0.6:
                    # Suppress the one with fewer frames occluded (newer anchor)
                    if a1.frames_occluded > a2.frames_occluded:
                        to_suppress.add(id2)
                        print(f"  [ANCHOR MERGE] Suppressed duplicate anchor {id2} in favor of older anchor {id1}.")
                    else:
                        to_suppress.add(id1)
                        print(f"  [ANCHOR MERGE] Suppressed duplicate anchor {id1} in favor of older anchor {id2}.")
                        
        for rid in to_suppress:
            del self.amodal_anchors[rid]
        
        # 6. Output unified track list
        outputs = []
        for track in self.baseline_tracker.tracks:
            if track.time_since_update == 0:
                outputs.append({
                    "track_id": track.track_id,
                    "bbox": track.last_bbox,
                    "status": "visible"
                })
                
        for track_id, anchor in self.amodal_anchors.items():
            x1 = anchor.cx - anchor.w / 2.0
            y1 = anchor.cy - anchor.h / 2.0
            x2 = anchor.cx + anchor.w / 2.0
            y2 = anchor.cy + anchor.h / 2.0
            outputs.append({
                "track_id": track_id,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "status": "occluded_virtual"
            })
            
        return outputs

    def _calculate_iou(self, box1, box2):
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

if __name__ == "__main__":
    from baseline_tracker import BoundingBoxTracker
    from projection import GroundPlaneProjector
    
    print("Testing OcclusionAwareTracker with Appearance Cues and Lifecycle Policies...")
    
    projector = GroundPlaneProjector()
    baseline = BoundingBoxTracker(max_lost_frames=30)
    tracker = OcclusionAwareTracker(baseline, projector)
    
    # 3-channel dummy images representing BGR frames
    frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame1[100:150, 100:150, 0] = 10 # low hue
    frame1[100:150, 100:150, 1] = 200
    frame1[100:150, 100:150, 2] = 200
    detections1 = [[100, 100, 150, 150]]
    
    # Target goes into occlusion area (green canopy)
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame2[80:200, 80:200, 1] = 200 # green mask
    detections2 = []
    
    # Target re-emerges with same appearance
    frame3 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame3[140:190, 220:270, 0] = 12
    frame3[140:190, 220:270, 1] = 195
    frame3[140:190, 220:270, 2] = 205
    detections3 = [[220, 140, 270, 190]]
    
    # Add static background dots to all frames to prevent registration drift
    for f in [frame1, frame2, frame3]:
        f[10, 10, :] = 255
        f[10, 500, :] = 255
        f[500, 10, :] = 255
        f[500, 500, :] = 255
        f[800, 800, :] = 255
        f[800, 10, :] = 255
        f[10, 800, :] = 255
    
    print("\n--- STEP 1: Animal Visible ---")
    res1 = tracker.step(frame1, detections1, 30.0, -90.0, 0.0)
    print("Results:", res1)
    
    print("\n--- STEP 2: Animal Occluded (Anchor Seeded) ---")
    res2 = tracker.step(frame2, detections2, 30.0, -90.0, 0.0)
    print("Results:", res2)
    
    # Force anchor motion towards emergence area
    tracker.amodal_anchors[1].velocity = np.array([171.43, 57.14], dtype=np.float32)
    
    print("\n--- STEP 3: Animal Re-emerges (Appearance Verified) ---")
    res3 = tracker.step(frame3, detections3, 30.0, -90.0, 0.0)
    print("Results:", res3)
    
    print("\n--- STEP 4: Testing Anchor Expiry ---")
    # Create a frame with a larger canopy mask that covers the target's current position (245, 165)
    frame4 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame4[80:400, 80:400, 1] = 200 # green mask
    
    # Add static background dots to prevent registration drift
    for f in [frame4]:
        f[10, 10, :] = 255
        f[10, 500, :] = 255
        f[500, 10, :] = 255
        f[500, 500, :] = 255
        f[800, 800, :] = 255
        f[800, 10, :] = 255
        f[10, 800, :] = 255
    
    # Add a mock anchor and force its frames_occluded past threshold
    tracker.step(frame4, [], 30.0, -90.0, 0.0) # seeds track 1 again in occlusion
    print("Seeded track 1 again in occlusion. Forcing occlusion duration...")
    tracker.amodal_anchors[1].frames_occluded = 150
    # Step again, should expire
    tracker.step(frame4, [], 30.0, -90.0, 0.0)
