import numpy as np
import cv2
from .track import CounterfactualAmodalTrack
from .motion import predict_track, update_track, apply_camera_compensation, compare_feat_similarity
from .association import associate_tracks, calculate_iou
from .social import calculate_herd_velocity
from .occlusion import CanopyOcclusionHandler
from .config import TrackerConfig

def calculate_social_similarity(track_or_det_box, old_track, mean_herd_vel):
    if mean_herd_vel is None or (abs(mean_herd_vel[0]) < 1e-3 and abs(mean_herd_vel[1]) < 1e-3):
        return None
        
    if isinstance(track_or_det_box, (list, np.ndarray)):
        det_box = track_or_det_box
    else:
        det_box = track_or_det_box.last_bbox
        
    det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
    det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
    
    exp_cx = old_track.motion["x"][0, 0] + mean_herd_vel[0]
    exp_cy = old_track.motion["x"][1, 0] + mean_herd_vel[1]
    
    dist_herd = np.hypot(det_cx - exp_cx, det_cy - exp_cy)
    s_soc = 1.0 - min(1.0, dist_herd / 100.0)
    return s_soc

class CounterfactualAmodalTracker:
    def __init__(self, projector, extractor, config=None):
        self.projector = projector
        self.extractor = extractor
        self.config = config if config is not None else TrackerConfig()
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.prev_frame = None
        self.occlusion_handler = CanopyOcclusionHandler()

    @property
    def next_id_val(self):
        return self.next_id

    @property
    def amodal_anchors(self):
        """
        Provides compatibility with legacy tools (like benchmark.py) by returning
        a dictionary of tracks currently in OCCLUDED or REMERGING state.
        """
        return {t.track_id: t for t in self.tracks if t.state in ["OCCLUDED", "REMERGING"]}

    def step(self, frame, detections, altitude=30.0, pitch=-90.0, roll=0.0, max_timeout=150, cohesion_weight=0.5, verbose=False):
        self.frame_count += 1
        if verbose:
            print("---------------------------------------------")
            print(f"[DIAGNOSTIC] === FRAME {self.frame_count} ===")
            for track in self.tracks:
                tsu = 0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"])
                print(f"[DIAGNOSTIC] Pre-Step: ID {track.track_id} | state={track.state} | tsu={tsu} | pos=({track.motion['x'][0, 0]:.1f}, {track.motion['x'][1, 0]:.1f})")

        # 1. Camera Motion Compensation (CMC)
        if self.prev_frame is not None and frame is not None:
            apply_camera_compensation(self.tracks, self.prev_frame, frame)
        self.prev_frame = frame if frame is None else frame.copy()

        # 2. Segment vegetation to identify occlusion zones
        occ_mask = None
        if frame is not None and self.projector is not None:
            h, w = frame.shape[:2]
            frame_small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            occ_mask = self.projector.segment_vegetation(frame_small)
            del frame_small

        # 3. Calculate visible herd mean velocity for social herd prior
        mean_herd_vel = calculate_herd_velocity(self.tracks)

        # 4. Motion Prediction
        predicted_bboxes = []
        for track in self.tracks:
            pred_box = predict_track(track, dt=1.0, herd_mean_vel=mean_herd_vel, cohesion_weight=cohesion_weight)
            predicted_bboxes.append(pred_box)

        # 5. Extract appearance features for detections
        det_features = []
        if len(detections) > 0 and frame is not None and self.extractor is not None:
            det_features = self.extractor.extract_features_batch(frame, detections)
        else:
            det_features = [None] * len(detections)

        # 6. Associate detections with tracks
        matches, unmatched_tracks, unmatched_dets = associate_tracks(
            self.tracks, detections, det_features, mean_herd_vel=mean_herd_vel, frame_count=self.frame_count, config=self.config, frame=frame, verbose=verbose
        )

        matched_track_indices = set()
        matched_det_indices = set()

        # Update matched tracks
        for track_idx, det_idx, match_cost in matches:
            track = self.tracks[track_idx]
            update_track(track, detections[det_idx], frame=frame, appearance_embedding=det_features[det_idx], frame_count=self.frame_count, match_cost=match_cost)
            track.identity["last_matched_frame"] = self.frame_count
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

        if verbose:
            for i, track in enumerate(self.tracks):
                matched = i in matched_track_indices
                tsu = 0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"])
                print(f"[DIAGNOSTIC] Post-Association: ID {track.track_id} | state={track.state} | matched={matched} | tsu={tsu}")

        # 7. Update unmatched tracks (State Transitions)
        for i, track in enumerate(self.tracks):
            if i in matched_track_indices:
                continue

            if track.state in ["VISIBLE", "NEW"]:
                # Check if target is in segmented canopy mask
                cx = float(track.motion["x"][0, 0])
                cy = float(track.motion["x"][1, 0])
                w = float(track.motion["x"][5, 0])
                h = float(track.motion["x"][6, 0])
                if self.occlusion_handler.is_occluded(cx, cy, occ_mask, w_box=w, h_box=h):
                    track.state = "OCCLUDED"
                    track.occlusion["entry_frame"] = self.frame_count
                else:
                    track.state = "LOST"
            elif track.state == "OCCLUDED":
                if track.occlusion["frames_occluded"] > 15:
                    track.state = "SEARCH"
            elif track.state == "LOST":
                time_lost = self.frame_count - track.identity["last_matched_frame"]
                if time_lost > 15:
                    track.state = "SEARCH"
            elif track.state == "SEARCH":
                time_lost = self.frame_count - track.identity["last_matched_frame"]
                vel_mag = float(track.motion["x"][2, 0])
                timeout_limit = max_timeout * 2 if vel_mag < 0.3 else max_timeout
                if time_lost > timeout_limit:
                    track.state = "EXPIRED"

        # Update/evaluate existing tentative tracks first
        to_suppress_reid = set()
        for track in self.tracks:
            if track.state == "TENTATIVE_REID":
                # Check if it was matched in this frame
                if track.track_id not in matched_track_indices:
                    # Discard this tentative track immediately
                    track.state = "EXPIRED"
                    continue
                
                hyp = track.identity_hypothesis
                old_track = next((t for t in self.tracks if t.track_id == hyp["old_id"]), None)
                if old_track is not None:
                    # Compute current frame Re-ID score
                    tsu = self.frame_count - old_track.identity["last_matched_frame"]
                    t_clamped = min(float(tsu), 120.0)
                    tau = self.config.dynamic_weight_tau
                    
                    # Adaptive Re-ID weights
                    w_app = 0.60 + 0.20 * (1.0 - np.exp(-t_clamped / tau))
                    w_beh = 0.20 - 0.05 * (1.0 - np.exp(-t_clamped / tau))
                    w_soc = 0.20 - 0.15 * (1.0 - np.exp(-t_clamped / tau))
                    
                    # Compute similarities
                    from .track import extract_texture_and_structural, extract_color_histogram
                    tex_new, struc_new = extract_texture_and_structural(frame, track.last_bbox)
                    det_col = extract_color_histogram(frame, track.last_bbox)
                    if det_col is None:
                        det_col = track.identity["appearance_embedding"]
                    det_ar = (track.last_bbox[2] - track.last_bbox[0]) / max(track.last_bbox[3] - track.last_bbox[1], 1e-6)
                    
                    avg_emb, avg_col = old_track.get_identity_prototype()
                    s_app = old_track.identity_memory.appearance.compute_similarity(
                        track.identity["appearance_embedding"],
                        det_col,
                        det_ar,
                        tex_new,
                        struc_new
                    )
                    
                    s_beh = 1.0 - min(1.0, abs(track.motion["x"][2, 0] - old_track.identity_memory.motion.average_speed) / max(1.0, old_track.identity_memory.motion.average_speed))
                    
                    s_soc = calculate_social_similarity(track, old_track, mean_herd_vel)
                    if s_soc is None:
                        w_sum = w_app + w_beh
                        score = (w_app / w_sum) * s_app + (w_beh / w_sum) * s_beh
                    else:
                        score = w_app * s_app + w_beh * s_beh + w_soc * s_soc
                    
                    hyp["scores"].append(score)
                    hyp["frames_buffered"] += 1
                    
                    if verbose:
                        print(f"[HYPOTHESIS BUFFER] Track {track.track_id} (tentative) match with {old_track.track_id} | Frames: {hyp['frames_buffered']}/{self.config.reid_buffer_length} | Score: {score:.3f}")
                    
                    if hyp["frames_buffered"] >= self.config.reid_buffer_length:
                        avg_score = sum(hyp["scores"]) / len(hyp["scores"])
                        if avg_score >= self.config.reid_min_average_score:
                            if verbose:
                                print(f"[REID SUCCESS] Confirmed recovery: Merging Track {track.track_id} into Old ID {old_track.track_id} (avg_score={avg_score:.3f})")
                            
                            # Merge track states
                            old_track.state = "VISIBLE"
                            old_track.motion["x"] = track.motion["x"].copy()
                            old_track.motion["P"] = track.motion["P"].copy()
                            old_track.identity["appearance_embedding"] = track.identity["appearance_embedding"]
                            old_track.identity["last_matched_frame"] = self.frame_count
                            
                            old_track.identity_memory.update(
                                track.identity["appearance_embedding"],
                                det_col,
                                det_ar,
                                float(track.motion["x"][2, 0]),
                                float(track.motion["x"][3, 0]),
                                avg_score,
                                recovered=True,
                                texture=tex_new,
                                structural=struc_new
                            )
                            old_track.last_updated_cx = track.last_updated_cx
                            old_track.last_updated_cy = track.last_updated_cy
                            old_track.last_bbox = track.last_bbox
                            
                            to_suppress_reid.add(track.track_id)
                        else:
                            if verbose:
                                print(f"[REID REJECT] Failed recovery match: Splitting Track {track.track_id} from Old ID {old_track.track_id} (avg_score={avg_score:.3f})")
                            # Promote tentative track to visible new identity
                            track.state = "VISIBLE"
                            track.persistent_identity_id = self.next_id
                            self.next_id += 1
                            track.identity_hypothesis = None
                else:
                    # Old track disappeared, promote tentative track
                    track.state = "VISIBLE"
                    track.persistent_identity_id = self.next_id
                    self.next_id += 1
                    track.identity_hypothesis = None

        # 8. Post-Association Re-ID Identity Recovery (Hypothesis Resurrection) for unmatched detections
        occluded_candidates = [t for t in self.tracks if t.state in ["OCCLUDED", "SEARCH", "LOST"]]
        
        unmatched_detections = list(unmatched_dets)
        matched_detections = list(matched_det_indices)
        if verbose:
            print(f"[DIAGNOSTIC] Detections count: {len(detections)}. Matched detections: {matched_detections}, Unmatched detections: {unmatched_detections}")

        for j in unmatched_dets:
            det_box = detections[j]
            det_feat = det_features[j]
            
            from .track import extract_texture_and_structural, extract_color_histogram
            tex_new, struc_new = extract_texture_and_structural(frame, det_box)
            det_col = extract_color_histogram(frame, det_box)
            if det_col is None:
                det_col = det_feat
            det_w = det_box[2] - det_box[0]
            det_h = det_box[3] - det_box[1]
            det_ar = det_w / max(det_h, 1e-6)
            det_cx = det_box[0] + det_w / 2.0
            det_cy = det_box[1] + det_h / 2.0
            
            best_score = -1.0
            best_old_track = None
            
            for old_track in occluded_candidates:
                tsu = self.frame_count - old_track.identity["last_matched_frame"]
                t_clamped = min(float(tsu), 120.0)
                tau = self.config.dynamic_weight_tau
                
                w_app = 0.60 + 0.20 * (1.0 - np.exp(-t_clamped / tau))
                w_beh = 0.20 - 0.05 * (1.0 - np.exp(-t_clamped / tau))
                w_soc = 0.20 - 0.15 * (1.0 - np.exp(-t_clamped / tau))
                
                avg_emb, avg_col = old_track.get_identity_prototype()
                s_app = old_track.identity_memory.appearance.compute_similarity(
                    det_feat,
                    det_col,
                    det_ar,
                    tex_new,
                    struc_new
                )
                
                # Estimate speed if matched to this old track
                dx = det_cx - old_track.motion["x"][0, 0]
                dy = det_cy - old_track.motion["x"][1, 0]
                dist = np.hypot(dx, dy)
                elapsed_frames = max(1.0, float(tsu))
                est_speed = dist / elapsed_frames
                
                s_beh = 1.0 - min(1.0, abs(est_speed - old_track.identity_memory.motion.average_speed) / max(1.0, old_track.identity_memory.motion.average_speed))
                
                s_soc = calculate_social_similarity(det_box, old_track, mean_herd_vel)
                
                if s_soc is None:
                    w_sum = w_app + w_beh
                    score = (w_app / w_sum) * s_app + (w_beh / w_sum) * s_beh
                else:
                    score = w_app * s_app + w_beh * s_beh + w_soc * s_soc
                    
                if score > best_score:
                    best_score = score
                    best_old_track = old_track
                    
            if best_old_track is not None:
                if best_score >= self.config.reid_merge_threshold:
                    if verbose:
                        print(f"[REID IMMEDIATE MERGE] Merging detection {j} into old Track {best_old_track.track_id} (score={best_score:.3f})")
                    
                    update_track(best_old_track, det_box, frame=frame, appearance_embedding=det_feat, frame_count=self.frame_count, match_cost=1.0 - best_score)
                    best_old_track.state = "VISIBLE"
                    best_old_track.identity["last_matched_frame"] = self.frame_count
                    
                    occluded_candidates.remove(best_old_track)
                    continue
                elif best_score >= self.config.reid_tentative_threshold:
                    if verbose:
                        print(f"[REID DEFERRED COMMITMENT] Deferring commitment for detection {j} with old Track {best_old_track.track_id} (score={best_score:.3f})")
                    
                    new_track = CounterfactualAmodalTrack(
                        det_box,
                        track_id=self.next_id,
                        frame=frame,
                        extractor=self.extractor,
                        appearance_embedding=det_feat,
                        persistent_identity_id=None
                    )
                    self.next_id += 1
                    new_track.state = "TENTATIVE_REID"
                    new_track.identity["last_matched_frame"] = self.frame_count
                    new_track.identity_hypothesis = {
                        "old_id": best_old_track.track_id,
                        "frames_buffered": 1,
                        "scores": [best_score]
                    }
                    self.tracks.append(new_track)
                    occluded_candidates.remove(best_old_track)
                    continue
            
            # If not matched or deferred, spawn new visible track
            if verbose:
                print(f"[TRACK BIRTH] Spawning new visible Track ID {self.next_id} for detection {j}")
            new_track = CounterfactualAmodalTrack(
                det_box,
                track_id=self.next_id,
                frame=frame,
                extractor=self.extractor,
                appearance_embedding=det_feat
            )
            self.next_id += 1
            new_track.state = "VISIBLE"
            new_track.identity["last_matched_frame"] = self.frame_count
            self.tracks.append(new_track)
            
        self.tracks = [t for t in self.tracks if t.track_id not in to_suppress_reid]

        # 9.5 Duplicate anchor suppression and merging
        to_suppress = set()
        for i in range(len(self.tracks)):
            for j in range(i + 1, len(self.tracks)):
                t1 = self.tracks[i]
                t2 = self.tracks[j]
                
                # Check bounding box overlap IoU
                box1 = t1._state_to_bbox(t1.motion["x"])
                box2 = t2._state_to_bbox(t2.motion["x"])
                iou = calculate_iou(box1, box2)
                
                if iou > 0.6:
                    # Check appearance similarity to ensure it's the same animal
                    avg_emb1, _ = t1.get_identity_prototype()
                    avg_emb2, _ = t2.get_identity_prototype()
                    sim = compare_feat_similarity(avg_emb1, avg_emb2)
                    if sim > 0.65:
                        older = t1 if t1.track_id < t2.track_id else t2
                        newer = t2 if t1.track_id < t2.track_id else t1
                        
                        # If the newer track is visible and the older track is occluded/lost,
                        # the older track inherits the state and position to prevent identity switches.
                        if newer.state in ["VISIBLE", "NEW"] and older.state not in ["VISIBLE", "NEW"]:
                            older.state = newer.state
                            older.motion["x"] = newer.motion["x"].copy()
                            older.motion["P"] = newer.motion["P"].copy()
                            older.identity["appearance_embedding"] = newer.identity["appearance_embedding"]
                            older.identity["last_matched_frame"] = newer.identity["last_matched_frame"]
                            older.last_updated_cx = newer.last_updated_cx
                            older.last_updated_cy = newer.last_updated_cy
                            
                        to_suppress.add(newer.track_id)
                            
        self.tracks = [t for t in self.tracks if t.track_id not in to_suppress]

        # 10. Update temporal memory and filter expired tracks
        active_tracks = []
        for track in self.tracks:
            if track.state == "EXPIRED":
                continue
            
            cx = float(track.motion["x"][0, 0])
            cy = float(track.motion["x"][1, 0])
            v = float(track.motion["x"][2, 0])
            theta = float(track.motion["x"][3, 0])
            vel = [v * np.cos(theta), v * np.sin(theta)]
            track.memory.update(cx, cy, track.state, self.frame_count, vel)
            
            active_tracks.append(track)
            
        self.tracks = active_tracks

        if verbose:
            for track in self.tracks:
                tsu = 0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"])
                print(f"[DIAGNOSTIC] Post-Cleanup: ID {track.track_id} | state={track.state} | tsu={tsu}")
            print(f"[DIAGNOSTIC] === END FRAME {self.frame_count} ===")

        # 11. Compile output structure
        outputs = []
        for track in self.tracks:
            bbox = track._state_to_bbox(track.motion["x"])
            v = float(track.motion["x"][2, 0])
            theta = float(track.motion["x"][3, 0])
            vel = [v * np.cos(theta), v * np.sin(theta)]
            
            if track.state in ["VISIBLE", "NEW"]:
                outputs.append({
                    "track_id": track.persistent_identity_id,
                    "bbox": bbox,
                    "status": "visible",
                    "velocity": vel
                })
            elif track.state in ["OCCLUDED", "REMERGING"]:
                # Flatten Sigma for outputs if needed or pass as list of lists
                sigma_list = track.motion["Sigma"].tolist()
                outputs.append({
                    "track_id": track.persistent_identity_id,
                    "bbox": bbox,
                    "status": "occluded_virtual",
                    "sigma": sigma_list,
                    "frames_occluded": track.occlusion["frames_occluded"],
                    "velocity": vel
                })
                
        return outputs
