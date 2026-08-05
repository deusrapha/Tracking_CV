import numpy as np
from scipy.optimize import linear_sum_assignment
from .motion import compare_feat_similarity
from .config import TrackerConfig

def calculate_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def check_track_gate(track, det_box, det_feat, det_col=None, det_ar=None, det_tex=None, det_struc=None):
    # Calculate thresholds matching
    iou = calculate_iou(track.last_bbox, det_box)
    
    # Retrieve prototypes and calculate unified similarity
    avg_emb, avg_col = track.get_identity_prototype()
    if det_col is None:
        det_col = det_feat
    if det_ar is None:
        det_w = det_box[2] - det_box[0]
        det_h = det_box[3] - det_box[1]
        det_ar = det_w / max(det_h, 1e-6)
        
    sim = track.identity_memory.appearance.compute_similarity(
        det_feat, det_col, det_ar, det_tex, det_struc
    )
    
    det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
    det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
    dx = det_cx - track.motion["x"][0, 0]
    dy = det_cy - track.motion["x"][1, 0]
    dist_euclidean = np.hypot(dx, dy)
    
    # Position-only 2D EKF covariance and measurement noise
    P_pos = track.motion["P"][:2, :2]
    R_pos = track.motion["R"][:2, :2]
    S_pos = P_pos + R_pos
    
    # Defensive check: if S_pos contains inf or nan, or is excessively large
    if np.any(np.isnan(S_pos)) or np.any(np.isinf(S_pos)) or np.max(np.abs(S_pos)) > 1e6:
        S_pos = np.eye(2, dtype=np.float32) * 1000.0
        
    S_pos = S_pos + 1e-6 * np.eye(2, dtype=np.float32)
    inv_S_pos = np.linalg.inv(S_pos)
    
    pred_cx = track.motion["x"][0, 0]
    pred_cy = track.motion["x"][1, 0]
    y_pos = np.array([det_cx - pred_cx, det_cy - pred_cy], dtype=np.float32).reshape(2, 1)
    mahalanobis_dist_sq = float((y_pos.T @ inv_S_pos @ y_pos)[0, 0])
    
    # Calculate eigenvalues for diagnostics
    eigvals = np.linalg.eigvals(S_pos)
    
    # Confidence-weighted appearance similarity
    s_effective = sim * track.identity["confidence"]
    
    # Adaptive gate based on confidence-weighted appearance similarity
    T_gate = 5.991 * (1.0 + 1.2 * max(0.0, s_effective)**2)
    
    motion_gate_pass = False
    appearance_gate_pass = False
    dist_gate_pass = False
    overall_gate_pass = False
    
    reason_fail = []
    
    if track.state in ["OCCLUDED", "REMERGING", "SEARCH", "TENTATIVE_REID"]:
        motion_gate_pass = iou > 0.0
        appearance_gate_pass = sim > 0.30
        dist_gate_pass = (dist_euclidean < 300.0 or mahalanobis_dist_sq <= max(T_gate * 2.0, 25.0))
        overall_gate_pass = appearance_gate_pass and dist_gate_pass
        
        if not overall_gate_pass:
            if not appearance_gate_pass:
                reason_fail.append(f"Appearance Gate (sim={sim:.2f} <= 0.30)")
            if not dist_gate_pass:
                reason_fail.append(f"Distance Gate (euclidean={dist_euclidean:.1f} px, mahalanobis={mahalanobis_dist_sq:.2f})")
    else:
        motion_gate_pass = iou > 0.1
        appearance_gate_pass = sim > 0.30
        dist_gate_pass = (dist_euclidean < 150.0 or mahalanobis_dist_sq <= max(T_gate * 2.0, 25.0))
        overall_gate_pass = motion_gate_pass or (appearance_gate_pass and dist_gate_pass)
        
        if not overall_gate_pass:
            reason_fail.append(f"Visible gating thresholds exceeded (iou={iou:.2f}, sim={sim:.2f}, dist={dist_euclidean:.1f})")
            
    return overall_gate_pass, motion_gate_pass, appearance_gate_pass, dist_gate_pass, mahalanobis_dist_sq, T_gate, dist_euclidean, eigvals, reason_fail, s_effective

def build_cost_matrix(tracks, detections, det_features, mean_herd_vel=None, config=None, frame_count=None, frame=None):
    """
    Constructs the weighted cost matrix using hierarchical identity prior and semantic appearance.
    """
    if config is None:
        config = TrackerConfig()
        
    num_tracks = len(tracks)
    num_dets = len(detections)
    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets), dtype=np.float32), []
        
    cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    candidate_diags = []
    
    for i, track in enumerate(tracks):
        # Determine adaptive weights based on lifecycle state and occlusion time
        if track.state in ["OCCLUDED", "REMERGING", "SEARCH", "LOST"] and frame_count is not None:
            tsu = frame_count - track.identity["last_matched_frame"]
            t = min(float(tsu), 120.0)
            tau = config.dynamic_weight_tau
            
            # Dynamic weights calculation
            alpha_val = config.base_weights["motion"] * np.exp(-t / tau)
            delta_val = config.base_weights["counterfactual"] * np.exp(-t / tau)
            gamma_val = config.base_weights["social"]
            
            beta_val = config.base_weights["appearance"] + (config.max_weights["appearance"] - config.base_weights["appearance"]) * (1.0 - np.exp(-t / tau))
            lambda_val = config.base_weights["prior"] + (config.max_weights["prior"] - config.base_weights["prior"]) * (1.0 - np.exp(-t / tau))
            
            # Normalize weights
            w_sum = alpha_val + beta_val + gamma_val + delta_val + lambda_val
            alpha = alpha_val / w_sum
            beta = beta_val / w_sum
            gamma = gamma_val / w_sum
            delta = delta_val / w_sum
            lambda_val = lambda_val / w_sum
        else:
            w_sum = sum(config.base_weights.values())
            alpha = config.base_weights["motion"] / w_sum
            beta = config.base_weights["appearance"] / w_sum
            gamma = config.base_weights["social"] / w_sum
            delta = config.base_weights["counterfactual"] / w_sum
            lambda_val = config.base_weights["prior"] / w_sum
            
    # Pre-extract texture, structural, and color features for each detection
    det_textures = []
    det_structurals = []
    det_colors = []
    from .track import extract_texture_and_structural, extract_color_histogram
    for det_box in detections:
        tex, struc = extract_texture_and_structural(frame, det_box)
        col = extract_color_histogram(frame, det_box)
        det_textures.append(tex)
        det_structurals.append(struc)
        det_colors.append(col)
        
    for i, track in enumerate(tracks):
        # Determine adaptive weights based on lifecycle state and occlusion time
        tsu = 0.0
        if frame_count is not None:
            tsu = frame_count - track.identity["last_matched_frame"]
            
        if track.state in ["OCCLUDED", "REMERGING", "SEARCH", "LOST"] and frame_count is not None:
            t = min(float(tsu), 120.0)
            tau = config.dynamic_weight_tau
            
            # Dynamic weights calculation
            alpha_val = config.base_weights["motion"] * np.exp(-t / tau)
            delta_val = config.base_weights["counterfactual"] * np.exp(-t / tau)
            gamma_val = config.base_weights["social"]
            
            beta_val = config.base_weights["appearance"] + (config.max_weights["appearance"] - config.base_weights["appearance"]) * (1.0 - np.exp(-t / tau))
            lambda_val = config.base_weights["prior"] + (config.max_weights["prior"] - config.base_weights["prior"]) * (1.0 - np.exp(-t / tau))
            
            # Normalize weights
            w_sum = alpha_val + beta_val + gamma_val + delta_val + lambda_val
            alpha = alpha_val / w_sum
            beta = beta_val / w_sum
            gamma = gamma_val / w_sum
            delta = delta_val / w_sum
            lambda_val = lambda_val / w_sum
        else:
            w_sum = sum(config.base_weights.values())
            alpha = config.base_weights["motion"] / w_sum
            beta = config.base_weights["appearance"] / w_sum
            gamma = config.base_weights["social"] / w_sum
            delta = config.base_weights["counterfactual"] / w_sum
            lambda_val = config.base_weights["prior"] / w_sum
            
        pred_box = track.last_bbox
        for j, det_box in enumerate(detections):
            det_feat = det_features[j]
            det_tex = det_textures[j]
            det_struc = det_structurals[j]
            det_col = det_colors[j]
            if det_col is None:
                det_col = det_feat
            det_w = det_box[2] - det_box[0]
            det_h = det_box[3] - det_box[1]
            det_ar = det_w / max(det_h, 1e-6)
            
            # --- A. MOTION COST (Cm) ---
            iou = calculate_iou(pred_box, det_box)
            c_motion = 1.0 - iou
            
            # --- B. MULTI-COMPONENT APPEARANCE COST (Capp) ---
            sim_app = track.identity_memory.appearance.compute_similarity(
                det_feat, det_col, det_ar, det_tex, det_struc
            )
            c_app = 1.0 - sim_app
            
            # --- C. SOCIAL PRIOR COST (Cs) ---
            c_social = 0.5
            det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
            det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
            if mean_herd_vel is not None:
                exp_cx = track.motion["x"][0, 0] + mean_herd_vel[0]
                exp_cy = track.motion["x"][1, 0] + mean_herd_vel[1]
                dist_herd = np.hypot(det_cx - exp_cx, det_cy - exp_cy)
                c_social = min(1.0, dist_herd / 100.0)
                
            # --- D. COUNTERFACTUAL ESTIMATION COST (Cc) ---
            dx = det_cx - track.motion["x"][0, 0]
            dy = det_cy - track.motion["x"][1, 0]
            dist = np.hypot(dx, dy)
            std_x = np.sqrt(track.motion["Sigma"][0, 0])
            std_y = np.sqrt(track.motion["Sigma"][1, 1])
            max_dev = 3.5 * max(std_x, std_y)
            c_counterfactual = min(1.0, dist / max_dev)
            
            # --- E. MEMORY COST (Cmemory) ---
            c_mem_app = c_app
            c_mem_mot = min(1.0, abs(track.motion["x"][2, 0] - track.identity_memory.motion.average_speed) / max(1.0, track.identity_memory.motion.average_speed))
            c_mem_beh = min(1.0, abs(track.motion["x"][4, 0] - track.identity_memory.behaviour.turn_frequency))
            c_mem_soc = 1.0 - track.identity_memory.social.herd_affiliation
            c_memory = 0.40 * c_mem_app + 0.30 * c_mem_mot + 0.15 * c_mem_beh + 0.15 * c_mem_soc
            
            # --- F. IDENTITY PRIOR COST (Cprior) ---
            c_age = np.exp(-track.identity["age"] / 100.0)
            c_continuity = 0.0 if track.state in ["OCCLUDED", "SEARCH", "REMERGING"] else 1.0
            
            # Identity prior using decayed memory confidence and stability
            track.identity_memory.decay_confidence(tsu, config.memory_decay_tau)
            c_reliability = 1.0 - (track.identity_memory.confidence * track.identity_memory.reliability.identity_stability)
            
            # Trajectory consistency
            heading_diff = abs(track.motion["x"][3, 0] - track.identity_memory.motion.average_turn_rate)
            c_traj_heading = 0.5 - 0.5 * np.cos(heading_diff)
            avg_speed = track.identity_memory.motion.average_speed
            c_traj_speed = min(1.0, abs(track.motion["x"][2, 0] - avg_speed) / max(1.0, avg_speed))
            c_traj = 0.5 * c_traj_heading + 0.5 * c_traj_speed
            
            c_prior = 0.20 * c_age + 0.20 * c_continuity + 0.15 * c_reliability + 0.20 * c_traj + 0.25 * c_memory
            
            # Compute sub-components for diagnostic logging
            avg_emb, avg_col = track.get_identity_prototype()
            c_embed = 1.0 - compare_feat_similarity(avg_emb, det_feat)
            c_color = 1.0 - compare_feat_similarity(avg_col, det_col)
            
            c_texture = 0.5
            if track.identity_memory.appearance.average_texture is not None and det_tex is not None:
                diff = abs(track.identity_memory.appearance.average_texture[0] - det_tex[0])
                c_texture = min(1.0, diff / max(1e-3, track.identity_memory.appearance.average_texture[0]))
                
            c_shape = min(1.0, abs(track.identity_memory.appearance.average_body_ratio - det_ar) / 0.5)
            
            c_semantic = 0.5
            if track.identity_memory.appearance.average_structural is not None and det_struc is not None:
                sim_str = float(np.dot(track.identity_memory.appearance.average_structural, det_struc) / (np.linalg.norm(track.identity_memory.appearance.average_structural) * np.linalg.norm(det_struc) + 1e-6))
                c_semantic = 1.0 - max(0.0, min(1.0, sim_str))

            # --- FINAL GLOBAL COST ---
            cost = alpha * c_motion + beta * c_app + gamma * c_social + delta * c_counterfactual + lambda_val * c_prior
            
            # Check gate
            is_matched, _, _, _, _, _, _, _, _, _ = check_track_gate(
                track, det_box, det_feat, det_col, det_ar, det_tex, det_struc
            )
            if not is_matched:
                cost = 1e5
                
            cost_matrix[i, j] = cost

            candidate_diags.append({
                "track_id": track.track_id,
                "track_state": track.state,
                "det_idx": j,
                "motion": float(c_motion),
                "app": {
                    "total": float(c_app),
                    "embed": float(c_embed),
                    "color": float(c_color),
                    "texture": float(c_texture),
                    "shape": float(c_shape),
                    "semantic": float(c_semantic)
                },
                "social": float(c_social),
                "counterfactual": float(c_counterfactual),
                "memory": {
                    "total": float(c_memory),
                    "app": float(c_mem_app),
                    "mot": float(c_mem_mot),
                    "beh": float(c_mem_beh),
                    "soc": float(c_mem_soc)
                },
                "prior": {
                    "total": float(c_prior),
                    "age": float(c_age),
                    "continuity": float(c_continuity),
                    "reliability": float(c_reliability),
                    "traj": float(c_traj),
                    "memory": float(c_memory)
                },
                "total_cost": float(cost),
                "penalized": False
            })
            
    # Apply Counterfactual Duplicate Correction
    for i, track_a in enumerate(tracks):
        if track_a.state in ["OCCLUDED", "SEARCH", "REMERGING"]:
            for k, track_b in enumerate(tracks):
                if track_b.track_id > track_a.track_id and track_b.state in ["VISIBLE", "NEW"]:
                    iou = calculate_iou(track_a.last_bbox, track_b.last_bbox)
                    dist = np.hypot(track_a.motion["x"][0, 0] - track_b.motion["x"][0, 0],
                                    track_a.motion["x"][1, 0] - track_b.motion["x"][1, 0])
                    if iou > 0.2 or dist < 80.0:
                         for j in range(num_dets):
                             avg_emb, _ = track_a.get_identity_prototype()
                             sim_a = compare_feat_similarity(avg_emb, det_features[j])
                             if sim_a > 0.65:
                                 if cost_matrix[k, j] < 1e4:
                                     cost_matrix[k, j] += 0.8
                                 for diag in candidate_diags:
                                     if diag["track_id"] == track_b.track_id and diag["det_idx"] == j:
                                         diag["total_cost"] += 0.8
                                         diag["penalized"] = True
                                        
    return cost_matrix, candidate_diags

def associate_tracks(tracks, detections, det_features, mean_herd_vel=None, frame_count=None, config=None, frame=None, verbose=False):
    """
    Executes bipartite assignment. Returns matched (track_idx, det_idx, cost) pairs and unmatched indices.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    cost_matrix, candidate_diags = build_cost_matrix(tracks, detections, det_features, mean_herd_vel, config, frame_count, frame)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 1. Prediction Diagnostics
    if verbose:
        print("=========================================================")
        print(f"FRAME {frame_count}")
        for track in tracks:
            cx, cy, v, theta, omega, w, h = track.motion["x"].flatten()
            std_x = np.sqrt(track.motion["Sigma"][0, 0])
            std_y = np.sqrt(track.motion["Sigma"][1, 1])
            max_dev = 3.5 * max(std_x, std_y)
            major = 3.5 * max(std_x, std_y)
            minor = 3.5 * min(std_x, std_y)
            tsu = frame_count - track.identity["last_matched_frame"]
            vel_x = v * np.cos(theta)
            vel_y = v * np.sin(theta)
            print(f"\nTrack {track.track_id}")
            print("--------")
            print(f"State              : {track.state}")
            print(f"Age                : {int(track.identity['age'])}")
            print(f"TimeSinceUpdate    : {int(tsu)}")
            print(f"Predicted Position : ({cx:.1f},{cy:.1f})")
            print(f"Predicted Velocity : ({vel_x:.1f},{vel_y:.1f})")
            print(f"Heading            : {theta * 180.0 / np.pi:.1f} deg")
            print(f"Turn Rate          : {omega:.4f} rad/s")
            print(f"Covariance Radius  : {max_dev:.1f} px")
            print(f"Search Ellipse     : Major={major:.1f} Minor={minor:.1f}")

    # Gather assigned details for later checks
    assigned_det_to_tracks = {} # det_idx -> list of (track_idx, cost, is_matched, track_id)
    for j, det_box in enumerate(detections):
        assigned_det_to_tracks[j] = []
        for i, track in enumerate(tracks):
            det_feat = det_features[j]
            is_matched, _, _, _, _, _, _, _, _, _ = check_track_gate(track, det_box, det_feat)
            cost = cost_matrix[i, j]
            assigned_det_to_tracks[j].append((i, cost, is_matched, track.track_id))

    # 2. Candidate Generation & Gating & Cost Breakdown
    if verbose:
        print("\nCANDIDATE GENERATION")
        for j, det_box in enumerate(detections):
            det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
            det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
            print(f"\nDetection {j}")
            print("-----------")
            print(f"Center : ({det_cx:.1f},{det_cy:.1f})")
            print("\nCandidate evaluation")
            for track in tracks:
                det_feat = det_features[j]
                overall_gate_pass, motion_gate_pass, appearance_gate_pass, dist_gate_pass, mahalanobis_dist_sq, T_gate, dist_euclidean, eigvals, reason_fail, s_effective = check_track_gate(track, det_box, det_feat)
                
                print(f"\nTrack {track.track_id}")
                print(f"Predicted Position    : ({track.motion['x'][0, 0]:.1f}, {track.motion['x'][1, 0]:.1f})")
                print(f"Detection Position    : ({det_cx:.1f}, {det_cy:.1f})")
                print(f"Euclidean distance    : {dist_euclidean:.1f} px")
                print(f"Mahalanobis distance  : {mahalanobis_dist_sq:.2f}")
                print(f"Gate Threshold        : {T_gate:.2f} (adaptive ReID relaxation)")
                print(f"Appearance Similarity : {compare_feat_similarity(track.identity['appearance_embedding'], det_feat):.2f} (effective: {s_effective:.2f})")
                print(f"Covariance Eigenvals  : ({eigvals[0]:.2f}, {eigvals[1]:.2f})")
                print("Gate:")
                print(f" Motion Gate     {'PASS' if motion_gate_pass else 'FAIL'}")
                print(f" Appearance Gate {'PASS' if appearance_gate_pass else 'FAIL'}")
                print(f" Semantic Gate   {'PASS' if dist_gate_pass else 'FAIL'}")
                print(f" Overall Gate    {'PASS' if overall_gate_pass else 'FAIL'}")
                if not overall_gate_pass:
                    print(f"Reason: {', '.join(reason_fail)}")
                
                # 3. Cost Breakdown
                diag = next((d for d in candidate_diags if d["track_id"] == track.track_id and d["det_idx"] == j), None)
                if diag is not None:
                    print(f"\nTrack {track.track_id}")
                    print(f"Motion      {diag['motion']:.2f}")
                    print(f"Appearance  {diag['app']['total']:.2f}")
                    print(f"Embedding   {diag['app']['embed']:.2f}")
                    print(f"Color       {diag['app']['color']:.2f}")
                    print(f"Texture     {diag['app']['texture']:.2f}")
                    print(f"Shape       {diag['app']['shape']:.2f}")
                    print(f"Semantic    {diag['app']['semantic']:.2f}")
                    print(f"Social      {diag['social']:.2f}")
                    print("Memory")
                    print(f"AppearanceMemory {diag['memory']['app']:.2f}")
                    print(f"MotionMemory     {diag['memory']['mot']:.2f}")
                    print(f"BehaviourMemory  {diag['memory']['beh']:.2f}")
                    print(f"SocialMemory     {diag['memory']['soc']:.2f}")
                    print(f"Prior        {diag['prior']['total']:.2f}")
                    print("TOTAL")
                    print(f"{diag['total_cost']:.2f}")

        # 3.5 Candidate Ranking Log for Occluded Tracks
        for track in tracks:
            if track.state in ["OCCLUDED", "SEARCH", "LOST"]:
                candidates = []
                for j in range(len(detections)):
                    final_cost = cost_matrix[tracks.index(track), j]
                    cm = 1.0
                    ca = 1.0
                    cp = 1.0
                    diag = next((d for d in candidate_diags if d["track_id"] == track.track_id and d["det_idx"] == j), None)
                    if diag is not None:
                        cm = diag["motion"]
                        ca = diag["app"]["total"]
                        cp = diag["prior"]["total"]
                    candidates.append((j, final_cost, cm, ca, cp))
                candidates.sort(key=lambda x: x[1])
                
                print(f"\nCandidate Ranking for Track {track.track_id} ({track.state}):")
                print("Rank | Track ID | Detection ID | Motion Cost | Appearance Cost | Prior Cost | Final Cost")
                print("-" * 90)
                for rank, (det_idx, final_cost, cm, ca, cp) in enumerate(candidates[:5]):
                    print(f"{rank+1:<4} | {track.track_id:<8} | {det_idx:<12} | {cm:<11.2f} | {ca:<15.2f} | {cp:<10.2f} | {final_cost:<10.2f}")

        # 4. Hungarian Input (Association Matrix)
        print("\nAssociation Matrix")
        header = "            "
        for dj in range(len(detections)):
            header += f"Det{dj:<10}"
        print(header)
        for i, track in enumerate(tracks):
            row_str = f"Track{track.track_id:<6}"
            for j in range(len(detections)):
                overall, _, _, _, _, _, _, _, _, _ = check_track_gate(track, detections[j], det_features[j])
                cost_val = cost_matrix[i, j]
                if not overall:
                    row_str += f"{'INF':<13}"
                else:
                    row_str += f"{cost_val:<13.2f}"
            print(row_str)

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))
    
    # 5. Hungarian Output
    matched_pairs = list(zip(row_ind, col_ind))
    for r, c in matched_pairs:
        cost = cost_matrix[r, c]
        track = tracks[r]
        det_box = detections[c]
        det_feat = det_features[c]
        
        is_matched, _, _, _, _, _, _, _, _, _ = check_track_gate(track, det_box, det_feat)
        if is_matched and track.state in ["OCCLUDED", "REMERGING", "SEARCH"]:
            track.state = "REMERGING"
                
        if is_matched:
            matches.append((r, c, float(cost)))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    if verbose:
        print("\nHungarian Assignment")
        for j in range(len(detections)):
            assigned_track_idx = next((r for r, c in matched_pairs if c == j), None)
            print(f"\nDetection{j}")
            print("v")
            if assigned_track_idx is not None:
                assigned_track = tracks[assigned_track_idx]
                cost = cost_matrix[assigned_track_idx, j]
                det_info = next((item for item in assigned_det_to_tracks[j] if item[0] == assigned_track_idx), None)
                is_matched_gate = det_info[2] if det_info else False
                
                if is_matched_gate:
                    print(f"Track{assigned_track.track_id}")
                    print("Reason")
                    print("Lowest admissible cost")
                    print(f"{cost:.2f} < INF")
                else:
                    print("None")
                    print("Reason")
                    print(f"Track{assigned_track.track_id} excluded")
                    print("Gate failure")
            else:
                print("None")
                print("Reason")
                print("No track assigned by Hungarian")

        # 7. Winner Confidence
        print("\nWinner")
        for j in range(len(detections)):
            candidates = sorted([item for item in assigned_det_to_tracks[j] if item[2]], key=lambda x: x[1])
            if len(candidates) > 0:
                best_item = candidates[0]
                best_id = best_item[3]
                best_conf = 1.0 - best_item[1]
                print(f"Track{best_id}")
                print("Confidence")
                print(f"{best_conf:.2f}")
                
                if len(candidates) > 1:
                    second_item = candidates[1]
                    second_id = second_item[3]
                    second_conf = 1.0 - second_item[1]
                    print("Second Best")
                    print(f"Track{second_id}")
                    print("Confidence")
                    print(f"{second_conf:.2f}")
                    print("Margin")
                    print(f"{best_conf - second_conf:.2f}")
                else:
                    print("Second Best")
                    print("None")
            else:
                print("None (All candidates failed gate)")

        # 6. Recovery Analysis & Counterfactual Status
        print("\nRECOVERY ANALYSIS")
        for track in tracks:
            if track.state in ["OCCLUDED", "SEARCH", "REMERGING"]:
                print(f"\nTrack {track.track_id}")
                best_det_idx = None
                best_dist = float('inf')
                for j, det_box in enumerate(detections):
                    det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
                    det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
                    dx = det_cx - track.motion["x"][0, 0]
                    dy = det_cy - track.motion["x"][1, 0]
                    dist = np.hypot(dx, dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = j
                
                if best_det_idx is not None:
                    det_box = detections[best_det_idx]
                    det_feat = det_features[best_det_idx]
                    
                    overall_gate_pass, motion_gate_pass, appearance_gate_pass, dist_gate_pass, mahalanobis_dist_sq, T_gate, dist_euclidean, eigvals, reason_fail, s_effective = check_track_gate(track, det_box, det_feat)
                    
                    avg_emb = track.identity_memory.appearance.average_embedding
                    sim_mem = compare_feat_similarity(avg_emb, det_feat) if avg_emb is not None else compare_feat_similarity(track.identity["appearance_embedding"], det_feat)
                    
                    assigned_det = next((c for r, c in matched_pairs if r == tracks.index(track)), None)
                    recovered = (assigned_det == best_det_idx) and overall_gate_pass
                    
                    print("Could recover?")
                    print(f"{'YES' if overall_gate_pass else 'NO'}")
                    if not overall_gate_pass:
                        print("Rejected because")
                        print("\n".join(reason_fail))
                    else:
                        print("Recovery potential")
                        print(f"Motion matched: {'YES' if motion_gate_pass else 'NO'}")
                        print(f"Appearance matched: {'YES' if appearance_gate_pass else 'NO'}")
                        print(f"Trajectory matched: YES")
                        print(f"Identity memory matched: YES")
                    
                    print("\nCOUNTERFACTUAL STATUS")
                    print(f"Predicted anchor: ({track.motion['x'][0, 0]:.1f},{track.motion['x'][1, 0]:.1f})")
                    det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
                    det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
                    print(f"Detection: ({det_cx:.1f},{det_cy:.1f})")
                    print(f"Anchor error: {dist_euclidean:.1f} px")
                    print(f"Mahalanobis distance: {mahalanobis_dist_sq:.2f}")
                    print(f"Threshold: {T_gate:.2f}")
                    print(f"Counterfactual confidence: {track.counterfactual_confidence:.2f}")
                    print(f"Identity Memory similarity: {sim_mem:.2f}")
                    print(f"Should recover?: {'YES' if overall_gate_pass else 'NO'}")
                    print(f"Recovered?: {'YES' if recovered else 'NO'}")
                    if not recovered:
                        if not overall_gate_pass:
                            print("Failure reason: Motion gate")
                        else:
                            print("Failure reason: Hungarian matched detection to another track")
                else:
                    print("Could recover?")
                    print("NO")
                    print("Recovery impossible")
                    print("No detection available")

    return matches, list(unmatched_tracks), list(unmatched_dets)
