import numpy as np
import cv2

def predict_track(track, dt=1.0, herd_mean_vel=None, cohesion_weight=0.5):
    """
    Predicts track motion using EKF CTRV or Counterfactual Herd Cohesion.
    """
    track.identity["age"] += dt
    
    # CTRV variables
    cx, cy, v, theta, omega, w, h = track.motion["x"].flatten()
    
    # Classify current behavior state mode
    if v < 0.2:
        track.behavior["mode"] = "stationary"
        q_scale = 0.05
    elif abs(omega) > 0.15:
        track.behavior["mode"] = "sharp turning"
        q_scale = 2.0
    elif v > 3.0:
        track.behavior["mode"] = "running"
        q_scale = 1.2
    else:
        track.behavior["mode"] = "walking"
        q_scale = 0.5
        
    if track.state in ["OCCLUDED", "REMERGING", "SEARCH", "LOST"]:
        track.occlusion["frames_occluded"] += dt
        
        # Growing uncertainty based on unobserved duration
        fo = track.occlusion["frames_occluded"]
        q_min = 1.5
        q_max = 8.0
        tau = 40.0
        q_scale = q_min + (q_max - q_min) * (1.0 - np.exp(-fo / tau))
            
        # Propagate heading using EKF turn rate omega
        theta += omega * dt
        track.motion["x"][3, 0] = theta
        
        # Herd Social Prior velocity blending
        self_vx = v * np.cos(theta)
        self_vy = v * np.sin(theta)
        blended_vx = self_vx
        blended_vy = self_vy
        if herd_mean_vel is not None:
            blended_vx = (1.0 - cohesion_weight) * self_vx + cohesion_weight * herd_mean_vel[0]
            blended_vy = (1.0 - cohesion_weight) * self_vy + cohesion_weight * herd_mean_vel[1]
            
        # Propagate counterfactual state position
        track.motion["x"][0, 0] += blended_vx * dt
        track.motion["x"][1, 0] += blended_vy * dt
        
        # Grow amodal uncertainty Sigma with behavior-scaled process noise
        track.motion["Sigma"] += track.motion["Q_sigma"] * q_scale * dt
        
        # Propagate state covariance P under occlusion
        F_j = np.eye(7, dtype=np.float32)
        if abs(omega) > 0.01:
            F_j[0, 2] = (np.sin(theta + omega * dt) - np.sin(theta)) / omega
            F_j[0, 3] = v * (np.cos(theta + omega * dt) - np.cos(theta)) / omega
            F_j[0, 4] = -v * (np.sin(theta + omega * dt) - np.sin(theta)) / (omega ** 2) + v * dt * np.cos(theta + omega * dt) / omega
            
            F_j[1, 2] = (-np.cos(theta + omega * dt) + np.cos(theta)) / omega
            F_j[1, 3] = v * (np.sin(theta + omega * dt) - np.sin(theta)) / omega
            F_j[1, 4] = -v * (-np.cos(theta + omega * dt) + np.cos(theta)) / (omega ** 2) + v * dt * np.sin(theta + omega * dt) / omega
        else:
            F_j[0, 2] = np.cos(theta) * dt
            F_j[0, 3] = -v * np.sin(theta) * dt
            F_j[1, 2] = np.sin(theta) * dt
            F_j[1, 3] = v * np.cos(theta) * dt
            
        F_j[3, 4] = dt
        track.motion["P"] = F_j @ track.motion["P"] @ F_j.T + track.motion["Q"] * q_scale * dt
        
        # Calculate dynamic counterfactual confidence score
        sigma_trace = float(track.motion["Sigma"][0, 0] + track.motion["Sigma"][1, 1])
        motion_conf = np.exp(-0.005 * max(0.0, sigma_trace - 10.0))
        appearance_conf = 0.995 ** track.occlusion["frames_occluded"]
        
        herd_conf = 1.0
        if herd_mean_vel is not None:
            norm_self = np.hypot(self_vx, self_vy)
            norm_herd = np.hypot(herd_mean_vel[0], herd_mean_vel[1])
            if norm_self > 0.1 and norm_herd > 0.1:
                cos_sim = (self_vx * herd_mean_vel[0] + self_vy * herd_mean_vel[1]) / (norm_self * norm_herd)
                herd_conf = 0.5 + 0.5 * max(0.0, cos_sim)
                
        terrain_conf = 0.9
        track.counterfactual_confidence = float(motion_conf * appearance_conf * herd_conf * terrain_conf)
    else:
        # Standard CTRV prediction using RK4
        def f_dot(state):
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
            
        k1 = f_dot(track.motion["x"].flatten())
        k2 = f_dot(track.motion["x"].flatten() + 0.5 * dt * k1)
        k3 = f_dot(track.motion["x"].flatten() + 0.5 * dt * k2)
        k4 = f_dot(track.motion["x"].flatten() + dt * k3)
        
        dx = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        track.motion["x"] = track.motion["x"] + dx.reshape(7, 1)
        
        # Clamp turn rate
        track.motion["x"][4, 0] = np.clip(track.motion["x"][4, 0], -0.5, 0.5)
        
        # Covariance Jacobian
        F_j = np.eye(7, dtype=np.float32)
        cur_v = float(track.motion["x"][2, 0])
        cur_theta = float(track.motion["x"][3, 0])
        cur_omega = float(track.motion["x"][4, 0])
        
        if abs(cur_omega) > 0.01:
            F_j[0, 2] = (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / cur_omega
            F_j[0, 3] = cur_v * (np.cos(cur_theta + cur_omega * dt) - np.cos(cur_theta)) / cur_omega
            F_j[0, 4] = -cur_v * (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / (cur_omega ** 2) + cur_v * dt * np.cos(cur_theta + cur_omega * dt) / cur_omega
            
            F_j[1, 2] = (-np.cos(cur_theta + cur_omega * dt) + np.cos(cur_theta)) / cur_omega
            F_j[1, 3] = cur_v * (np.sin(cur_theta + cur_omega * dt) - np.sin(cur_theta)) / cur_omega
            F_j[1, 4] = -cur_v * (-np.cos(cur_theta + cur_omega * dt) + np.cos(cur_theta)) / (cur_omega ** 2) + cur_v * dt * np.sin(cur_theta + cur_omega * dt) / cur_omega
        else:
            F_j[0, 2] = np.cos(cur_theta) * dt
            F_j[0, 3] = -cur_v * np.sin(cur_theta) * dt
            F_j[1, 2] = np.sin(cur_theta) * dt
            F_j[1, 3] = cur_v * np.cos(cur_theta) * dt
            
        F_j[3, 4] = dt
        track.motion["P"] = F_j @ track.motion["P"] @ F_j.T + track.motion["Q"] * q_scale * dt
        
        track.counterfactual_confidence = 1.0
        
    track.last_bbox = track._state_to_bbox(track.motion["x"])
    return track.last_bbox

def update_track(track, bbox, frame=None, appearance_embedding=None, frame_count=None, match_cost=0.0):
    """
    Updates the CTRV EKF state with observed bounding box.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    
    # Calculate physical displacement from last matched detection centroid
    dx_phys = cx - track.last_updated_cx
    dy_phys = cy - track.last_updated_cy
    dist_phys = float(np.hypot(dx_phys, dy_phys))
    
    # Calculate elapsed frames since last successful match
    elapsed_frames = 1
    if frame_count is not None and track.identity["last_matched_frame"] > 0:
        elapsed_frames = max(1, frame_count - track.identity["last_matched_frame"])
        
    if dist_phys > 1.0:
        new_heading = np.arctan2(dy_phys, dx_phys)
        new_speed = dist_phys / elapsed_frames
        
        # Append raw estimates to behavior history
        track.behavior["heading_history"].append(new_heading)
        track.behavior["speed_history"].append(new_speed)
        
        if len(track.behavior["heading_history"]) > 8:
            track.behavior["heading_history"].pop(0)
        if len(track.behavior["speed_history"]) > 8:
            track.behavior["speed_history"].pop(0)
            
        # Calculate smoothed turn rate (omega) from circular heading differences
        history = track.behavior["heading_history"]
        if len(history) >= 2:
            diffs = []
            for idx in range(len(history) - 1):
                diff = history[idx + 1] - history[idx]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                diffs.append(diff)
            omega = float(np.mean(diffs))
        else:
            omega = 0.0
            
        track.motion["x"][3, 0] = new_heading
        track.motion["x"][2, 0] = new_speed
        track.motion["x"][4, 0] = omega
    else:
        track.motion["x"][2, 0] *= 0.8  # Decelerate if stationary
        track.motion["x"][4, 0] *= 0.8  # Decelerate turn rate if stationary
        
        # Track speed decay in behavior history
        track.behavior["speed_history"].append(float(track.motion["x"][2, 0]))
        if len(track.behavior["speed_history"]) > 8:
            track.behavior["speed_history"].pop(0)
        
    z = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
    
    # EKF update step
    H = np.zeros((4, 7), dtype=np.float32)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 5] = 1.0
    H[3, 6] = 1.0
    
    S = H @ track.motion["P"] @ H.T + track.motion["R"]
    K = track.motion["P"] @ H.T @ np.linalg.inv(S)
    
    track.motion["x"] = track.motion["x"] + K @ (z - H @ track.motion["x"])
    track.motion["P"] = (np.eye(7, dtype=np.float32) - K @ H) @ track.motion["P"]
    
    # Update last updated coordinates to the new post-update position
    track.last_updated_cx = float(track.motion["x"][0, 0])
    track.last_updated_cy = float(track.motion["x"][1, 0])
        
    # Reset tracking covariance to reflect baseline uncertainty post-match
    track.motion["P"][0, 0] = 5.0
    track.motion["P"][1, 1] = 5.0
    track.motion["P"][5, 5] = 5.0
    track.motion["P"][6, 6] = 5.0
    
    # Collapse spatial amodal uncertainty Sigma
    track.motion["Sigma"] = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    
    # Reset occlusion frames count
    track.occlusion["frames_occluded"] = 0
    
    was_amodal = track.state in ["OCCLUDED", "REMERGING", "SEARCH"]
    track.state = "VISIBLE"
    
    # EMA appearance update
    new_feat = appearance_embedding
    if new_feat is None and frame is not None and track.extractor is not None:
        new_feat = track.extractor.extract_features(frame, bbox)
        
    cur_ar = w / h
    current_speed = float(track.motion["x"][2, 0])
    current_omega = float(track.motion["x"][4, 0])
    
    # Identity Memory Update Rule: Freeze memory except reliability if match confidence is low
    match_confidence = 1.0 - match_cost
    freeze = match_confidence <= 0.65
    
    from .track import extract_texture_and_structural, extract_color_histogram
    texture, structural = extract_texture_and_structural(frame, bbox)
    color_hist = extract_color_histogram(frame, bbox)
    if color_hist is None:
        color_hist = new_feat
        
    track.identity_memory.update(
        new_feat, 
        color_hist, 
        cur_ar, 
        current_speed, 
        current_omega, 
        match_confidence, 
        was_amodal,
        freeze=freeze,
        texture=texture,
        structural=structural
    )
        
    if new_feat is not None and not freeze:
        if track.identity["appearance_embedding"] is None:
            track.identity["appearance_embedding"] = new_feat
        else:
            sim = compare_feat_similarity(track.identity["appearance_embedding"], new_feat)
            if sim > 0.65:
                track.identity["appearance_embedding"] = 0.9 * track.identity["appearance_embedding"] + 0.1 * new_feat

def compare_feat_similarity(f1, f2):
    if f1 is None or f2 is None:
        return 0.5
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6))

def apply_camera_compensation(tracks, prev_frame, curr_frame):
    """
    Applies ORB-based Camera Motion Compensation (CMC) on downsampled frames to save memory.
    """
    if prev_frame is None or curr_frame is None:
        return
        
    # Scale down by 2x for memory and processing efficiency
    scale = 2.0
    h, w = prev_frame.shape[:2]
    new_w, new_h = int(w / scale), int(h / scale)
    
    prev_small = cv2.resize(prev_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    curr_small = cv2.resize(curr_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    
    del prev_small, curr_small
    
    orb = cv2.ORB_create(nfeatures=250)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    del prev_gray, curr_gray
    
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    
    if len(matches) < 6:
        return
        
    # Scale keypoint positions back up to match original frame dimensions
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) * scale
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) * scale
    
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        return
        
    theta_cam = np.arctan2(M[1, 0], M[0, 0])
    
    for track in tracks:
        cx = float(track.motion["x"][0, 0])
        cy = float(track.motion["x"][1, 0])
        
        # Warp position
        new_pos = M @ np.array([cx, cy, 1.0], dtype=np.float32).reshape(3, 1)
        track.motion["x"][0, 0] = new_pos[0, 0]
        track.motion["x"][1, 0] = new_pos[1, 0]
        
        # Warp heading angle
        track.motion["x"][3, 0] += theta_cam
        
        # Warp last updated coordinates to maintain alignment for physical velocity updates
        new_last = M @ np.array([track.last_updated_cx, track.last_updated_cy, 1.0], dtype=np.float32).reshape(3, 1)
        track.last_updated_cx = float(new_last[0, 0])
        track.last_updated_cy = float(new_last[1, 0])
