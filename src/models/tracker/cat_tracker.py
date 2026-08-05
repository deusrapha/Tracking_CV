import numpy as np
import cv2
from .track import CounterfactualAmodalTrack
from .motion import predict_track, update_track, apply_camera_compensation, compare_feat_similarity
from .association import associate_tracks, calculate_iou, compute_identity_link_score
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

def get_origin_type(bbox, img_w=1920, img_h=1080):
    x1, y1, x2, y2 = bbox
    margin = 15
    if x1 <= margin or y1 <= margin or x2 >= (img_w - margin) or y2 >= (img_h - margin):
        return "SCENE_ENTRY"
    return "OCCLUSION_REEMERGENCE"

class CounterfactualAmodalTracker:
    def __init__(self, projector, extractor, config=None):
        self.projector = projector
        self.extractor = extractor
        self.config = config if config is not None else TrackerConfig()
        self.tracks = []
        self.next_track_instance_id = 1
        self.next_cattle_id = 1
        self.cattle_identities = {} # cattle_id -> dict details
        self.frame_count = 0
        self.prev_frame = None
        self.occlusion_handler = CanopyOcclusionHandler()

    @property
    def next_id(self):
        return self.next_track_instance_id

    @next_id.setter
    def next_id(self, val):
        self.next_track_instance_id = val

    @property
    def next_id_val(self):
        return self.next_track_instance_id

    @property
    def amodal_anchors(self):
        """
        Returns a dictionary of tracks currently in hidden persistent states:
        OCCLUDED, SEARCH, REMERGING, or LOST.
        """
        return {t.track_instance_id: t for t in self.tracks if t.state in ["OCCLUDED", "SEARCH", "REMERGING", "LOST"]}

    def step(self, frame, detections, altitude=30.0, pitch=-90.0, roll=0.0, max_timeout=150, cohesion_weight=0.5, verbose=False):
        self.frame_count += 1
        img_h, img_w = (frame.shape[:2]) if frame is not None else (1080, 1920)

        if verbose:
            print("---------------------------------------------")
            print(f"[DIAGNOSTIC] === FRAME {self.frame_count} ===")
            for track in self.tracks:
                tsu = 0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"])
                print(f"[DIAGNOSTIC] Pre-Step: TrackInst {track.track_instance_id} (CattleID {track.cattle_id}) | state={track.state} | tsu={tsu} | pos=({track.motion['x'][0, 0]:.1f}, {track.motion['x'][1, 0]:.1f})")

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

        # Filter out SUPERSEDED tracks from frame-level Hungarian matching candidates
        active_candidates = [t for t in self.tracks if t.state != "SUPERSEDED"]
        candidate_to_global_idx = {i: self.tracks.index(t) for i, t in enumerate(active_candidates)}

        # 6. Associate detections with active tracks
        matches_active, unmatched_track_indices_active, unmatched_dets = associate_tracks(
            active_candidates, detections, det_features, mean_herd_vel=mean_herd_vel, frame_count=self.frame_count, config=self.config, frame=frame, verbose=verbose
        )

        matched_track_indices = set()
        matched_det_indices = set()
        matched_track_instance_ids = set()

        # Update matched tracks
        for cand_idx, det_idx, match_cost in matches_active:
            global_idx = candidate_to_global_idx[cand_idx]
            track = self.tracks[global_idx]
            matched_track_indices.add(global_idx)
            matched_det_indices.add(det_idx)
            matched_track_instance_ids.add(track.track_instance_id)

            if track.state in ["OCCLUDED", "SEARCH", "LOST", "REMERGING"]:
                # Re-emergence after occlusion: spawn a new TrackInstance segment, retain cattle_id, supersede old segment
                new_track = CounterfactualAmodalTrack(
                    detections[det_idx],
                    track_instance_id=self.next_track_instance_id,
                    cattle_id=track.cattle_id,
                    frame=frame,
                    extractor=self.extractor,
                    appearance_embedding=det_features[det_idx],
                    origin_type=get_origin_type(detections[det_idx], img_w=img_w, img_h=img_h)
                )
                self.next_track_instance_id += 1
                update_track(new_track, detections[det_idx], frame=frame, appearance_embedding=det_features[det_idx], frame_count=self.frame_count, match_cost=match_cost)
                new_track.state = "VISIBLE"
                new_track.predecessor_track_instance_id = track.track_instance_id
                new_track.identity["last_matched_frame"] = self.frame_count

                track.state = "SUPERSEDED"
                track.successor_track_instance_id = new_track.track_instance_id

                if new_track.cattle_id in self.cattle_identities:
                    self.cattle_identities[new_track.cattle_id]["active_track_instance_id"] = new_track.track_instance_id
                    self.cattle_identities[new_track.cattle_id]["historical_track_instance_ids"].append(new_track.track_instance_id)

                self.tracks.append(new_track)
                matched_track_instance_ids.add(new_track.track_instance_id)
            else:
                update_track(track, detections[det_idx], frame=frame, appearance_embedding=det_features[det_idx], frame_count=self.frame_count, match_cost=match_cost)
                track.identity["last_matched_frame"] = self.frame_count

        if verbose:
            for i, track in enumerate(self.tracks):
                matched = track.track_instance_id in matched_track_instance_ids
                tsu = 0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"])
                print(f"[DIAGNOSTIC] Post-Association: TrackInst {track.track_instance_id} (CattleID {track.cattle_id}) | state={track.state} | matched={matched} | tsu={tsu}")

        # 7. Update unmatched tracks (State Transitions)
        for i, track in enumerate(self.tracks):
            if track.track_instance_id in matched_track_instance_ids or track.state in ["SUPERSEDED", "EXPIRED", "TENTATIVE_REID"]:
                continue

            if track.state in ["VISIBLE", "NEW"]:
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
                if self.tracks.index(track) not in matched_track_indices:
                    track.state = "EXPIRED"
                    continue
                
                hyp = track.identity_hypothesis
                old_track = next((t for t in self.tracks if t.track_instance_id == hyp["old_id"]), None)
                if old_track is not None:
                    from .track import extract_texture_and_structural, extract_color_histogram
                    tex_new, struc_new = extract_texture_and_structural(frame, track.last_bbox)
                    det_col = extract_color_histogram(frame, track.last_bbox)
                    if det_col is None:
                        det_col = track.identity["appearance_embedding"]
                    det_ar = (track.last_bbox[2] - track.last_bbox[0]) / max(track.last_bbox[3] - track.last_bbox[1], 1e-6)

                    score = compute_identity_link_score(
                        old_track, track.last_bbox, track.identity["appearance_embedding"],
                        det_col=det_col, det_ar=det_ar, det_tex=tex_new, det_struc=struc_new,
                        mean_herd_vel=mean_herd_vel, frame_count=self.frame_count
                    )

                    hyp["scores"].append(score)
                    hyp["frames_buffered"] += 1
                    
                    if verbose:
                        print(f"[HYPOTHESIS BUFFER] TrackInst {track.track_instance_id} (tentative) match with OldInst {old_track.track_instance_id} (CattleID {old_track.cattle_id}) | Frames: {hyp['frames_buffered']}/{self.config.reid_buffer_length} | Score: {score:.3f}")
                    
                    if hyp["frames_buffered"] >= self.config.reid_buffer_length:
                        avg_score = sum(hyp["scores"]) / len(hyp["scores"])
                        if avg_score >= self.config.reid_min_average_score:
                            if verbose:
                                print(f"[REID SUCCESS] Confirmed recovery: Segment linking TrackInst {track.track_instance_id} to OldInst {old_track.track_instance_id} under CattleID {old_track.cattle_id} (score={avg_score:.3f})")
                            
                            # Link segments cleanly
                            old_track.state = "SUPERSEDED"
                            old_track.successor_track_instance_id = track.track_instance_id
                            track.predecessor_track_instance_id = old_track.track_instance_id
                            track.cattle_id = old_track.cattle_id
                            track.state = "VISIBLE"
                            track.identity_hypothesis = None

                            if track.cattle_id in self.cattle_identities:
                                self.cattle_identities[track.cattle_id]["active_track_instance_id"] = track.track_instance_id
                                self.cattle_identities[track.cattle_id]["historical_track_instance_ids"].append(track.track_instance_id)
                        else:
                            if verbose:
                                print(f"[REID REJECT] Failed recovery match: Allocating new Cattle ID for TrackInst {track.track_instance_id}")
                            track.state = "VISIBLE"
                            track.cattle_id = self.next_cattle_id
                            self.next_cattle_id += 1
                            track.identity_hypothesis = None
                            self.cattle_identities[track.cattle_id] = {
                                "active_track_instance_id": track.track_instance_id,
                                "historical_track_instance_ids": [track.track_instance_id],
                                "state": "ACTIVE"
                            }
                else:
                    track.state = "VISIBLE"
                    track.cattle_id = self.next_cattle_id
                    self.next_cattle_id += 1
                    track.identity_hypothesis = None
                    self.cattle_identities[track.cattle_id] = {
                        "active_track_instance_id": track.track_instance_id,
                        "historical_track_instance_ids": [track.track_instance_id],
                        "state": "ACTIVE"
                    }

        # 8. Post-Association Re-ID Identity Recovery for unmatched detections
        # Candidate eligibility for recovery: OCCLUDED, SEARCH, REMERGING, or LOST within lost_recovery_window (75)
        recoverable_candidates = [
            t for t in self.tracks
            if t.state in ["OCCLUDED", "SEARCH", "REMERGING"] or (
                t.state == "LOST" and
                (self.frame_count - t.identity["last_matched_frame"]) <= 75 and
                t.identity.get("confidence", 1.0) >= 0.40
            )
        ]
        
        unmatched_detections = list(unmatched_dets)
        matched_detections = list(matched_det_indices)
        if verbose:
            print(f"[DIAGNOSTIC] Detections count: {len(detections)}. Matched: {matched_detections}, Unmatched: {unmatched_detections}")

        for j in unmatched_dets:
            det_box = detections[j]
            det_feat = det_features[j]
            origin_type = get_origin_type(det_box, img_w=img_w, img_h=img_h)
            
            from .track import extract_texture_and_structural, extract_color_histogram
            tex_new, struc_new = extract_texture_and_structural(frame, det_box)
            det_col = extract_color_histogram(frame, det_box)
            if det_col is None:
                det_col = det_feat
            det_w = det_box[2] - det_box[0]
            det_h = det_box[3] - det_box[1]
            det_ar = det_w / max(det_h, 1e-6)
            
            best_score = -1.0
            best_old_track = None
            
            for old_track in recoverable_candidates:
                det_cx = det_box[0] + (det_box[2] - det_box[0]) / 2.0
                det_cy = det_box[1] + (det_box[3] - det_box[1]) / 2.0
                comp = old_track.occlusion.get("component_mask")
                if comp is not None and not self.occlusion_handler.is_point_inside_component(det_cx, det_cy, comp, margin=30.0):
                    continue

                score = compute_identity_link_score(
                    old_track, det_box, det_feat,
                    det_col=det_col, det_ar=det_ar, det_tex=tex_new, det_struc=struc_new,
                    mean_herd_vel=mean_herd_vel, frame_count=self.frame_count
                )
                if score > best_score:
                    best_score = score
                    best_old_track = old_track

            # Single Cattle Prior: In single-cattle scene with interior re-emergence, boost score
            existing_cattle_ids = {t.cattle_id for t in self.tracks if t.state in ["VISIBLE", "OCCLUDED", "SEARCH", "REMERGING", "LOST"]}
            if len(existing_cattle_ids) == 1 and best_old_track is not None and origin_type != "SCENE_ENTRY":
                if best_score >= 0.40:
                    best_score = max(best_score, self.config.reid_merge_threshold + 0.05)

            if best_old_track is not None:
                if best_score >= self.config.reid_merge_threshold:
                    if verbose:
                        print(f"[REID IMMEDIATE MERGE] Linking detection {j} to CattleID {best_old_track.cattle_id} (OldInst {best_old_track.track_instance_id}, score={best_score:.3f})")
                    
                    new_track = CounterfactualAmodalTrack(
                        det_box,
                        track_instance_id=self.next_track_instance_id,
                        cattle_id=best_old_track.cattle_id,
                        frame=frame,
                        extractor=self.extractor,
                        appearance_embedding=det_feat,
                        origin_type=origin_type
                    )
                    self.next_track_instance_id += 1
                    new_track.state = "VISIBLE"
                    new_track.predecessor_track_instance_id = best_old_track.track_instance_id
                    new_track.identity["last_matched_frame"] = self.frame_count
                    
                    best_old_track.state = "SUPERSEDED"
                    best_old_track.successor_track_instance_id = new_track.track_instance_id
                    
                    if new_track.cattle_id in self.cattle_identities:
                        self.cattle_identities[new_track.cattle_id]["active_track_instance_id"] = new_track.track_instance_id
                        self.cattle_identities[new_track.cattle_id]["historical_track_instance_ids"].append(new_track.track_instance_id)

                    self.tracks.append(new_track)
                    recoverable_candidates.remove(best_old_track)
                    continue
                elif best_score >= self.config.reid_tentative_threshold:
                    if verbose:
                        print(f"[REID DEFERRED COMMITMENT] Deferring commitment for detection {j} with CattleID {best_old_track.cattle_id} (score={best_score:.3f})")
                    
                    new_track = CounterfactualAmodalTrack(
                        det_box,
                        track_instance_id=self.next_track_instance_id,
                        cattle_id=best_old_track.cattle_id,
                        frame=frame,
                        extractor=self.extractor,
                        appearance_embedding=det_feat,
                        origin_type=origin_type
                    )
                    self.next_track_instance_id += 1
                    new_track.state = "TENTATIVE_REID"
                    new_track.identity["last_matched_frame"] = self.frame_count
                    new_track.identity_hypothesis = {
                        "old_id": best_old_track.track_instance_id,
                        "frames_buffered": 1,
                        "scores": [best_score]
                    }
                    self.tracks.append(new_track)
                    recoverable_candidates.remove(best_old_track)
                    continue

            assigned_cattle_id = self.next_cattle_id
            self.next_cattle_id += 1
            active_cattle = [cid for cid, rec in self.cattle_identities.items() if rec.get("state") == "ACTIVE"]
            searching_cattle = [cid for cid, rec in self.cattle_identities.items() if rec.get("state") in ["SEARCHING", "SEARCH"]]
            inactive_cattle = [cid for cid, rec in self.cattle_identities.items() if rec.get("state") in ["INACTIVE_SEARCH", "INACTIVE"] and cid not in active_cattle]
            historical_tracks = list({t.track_instance_id for t in self.tracks if t.state in ["SUPERSEDED", "EXPIRED"]})
            if verbose:
                print(f"[IDENTITY REGISTRY] ActiveCattle={active_cattle}, SearchingCattle={searching_cattle}, InactiveCattle={inactive_cattle}, HistoricalTracks={historical_tracks}")
                print(f"[TRACK BIRTH] Allocating new Cattle ID {assigned_cattle_id} (TrackInst {self.next_track_instance_id}) for detection {j} (origin={origin_type})")
            
            new_track = CounterfactualAmodalTrack(
                det_box,
                track_instance_id=self.next_track_instance_id,
                cattle_id=assigned_cattle_id,
                frame=frame,
                extractor=self.extractor,
                appearance_embedding=det_feat,
                origin_type=origin_type
            )
            self.next_track_instance_id += 1
            new_track.state = "VISIBLE"
            new_track.identity["last_matched_frame"] = self.frame_count
            self.tracks.append(new_track)
            
            self.cattle_identities[assigned_cattle_id] = {
                "active_track_instance_id": new_track.track_instance_id,
                "historical_track_instance_ids": [new_track.track_instance_id],
                "state": "ACTIVE"
            }
            
        self.tracks = [t for t in self.tracks if t.track_instance_id not in to_suppress_reid]

        # 9.5 Duplicate anchor suppression and merging
        to_suppress = set()
        for i in range(len(self.tracks)):
            for j in range(i + 1, len(self.tracks)):
                t1 = self.tracks[i]
                t2 = self.tracks[j]
                if t1.state in ["SUPERSEDED", "EXPIRED"] or t2.state in ["SUPERSEDED", "EXPIRED"]:
                    continue
                
                box1 = t1._state_to_bbox(t1.motion["x"])
                box2 = t2._state_to_bbox(t2.motion["x"])
                iou = calculate_iou(box1, box2)
                
                if iou > 0.6:
                    avg_emb1, _ = t1.get_identity_prototype()
                    avg_emb2, _ = t2.get_identity_prototype()
                    sim = compare_feat_similarity(avg_emb1, avg_emb2)
                    if sim > 0.65:
                        older = t1 if t1.track_instance_id < t2.track_instance_id else t2
                        newer = t2 if t1.track_instance_id < t2.track_instance_id else t1
                        
                        if newer.state in ["VISIBLE", "NEW"] and older.state not in ["VISIBLE", "NEW"]:
                            older.state = "SUPERSEDED"
                            older.successor_track_instance_id = newer.track_instance_id
                            newer.predecessor_track_instance_id = older.track_instance_id
                            newer.cattle_id = older.cattle_id
                        elif older.state in ["VISIBLE", "NEW"] and newer.state in ["VISIBLE", "NEW"]:
                            newer.state = "SUPERSEDED"
                            newer.successor_track_instance_id = older.track_instance_id
                            to_suppress.add(newer.track_instance_id)
                            
        self.tracks = [t for t in self.tracks if t.track_instance_id not in to_suppress]

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
                print(f"[DIAGNOSTIC] Post-Cleanup: TrackInst {track.track_instance_id} | CattleID {track.cattle_id} | state={track.state} | tsu={tsu}")
            print(f"[DIAGNOSTIC] === END FRAME {self.frame_count} ===")

        # 11. Compile separate output structures for visible tracks vs amodal anchors
        outputs = []
        for track in self.tracks:
            bbox = track._state_to_bbox(track.motion["x"])
            v = float(track.motion["x"][2, 0])
            theta = float(track.motion["x"][3, 0])
            vel = [v * np.cos(theta), v * np.sin(theta)]
            
            tsu = getattr(track, 'time_since_update', 0) if not hasattr(track, 'identity') else (0 if track.state in ["VISIBLE", "NEW"] else (self.frame_count - track.identity["last_matched_frame"]))

            if track.state in ["VISIBLE", "NEW"] and tsu == 0:
                outputs.append({
                    "track_id": track.cattle_id, # displayed Cattle ID
                    "track_instance_id": track.track_instance_id,
                    "cattle_id": track.cattle_id,
                    "bbox": bbox,
                    "status": "visible",
                    "velocity": vel
                })
            elif track.state in ["OCCLUDED", "SEARCH", "REMERGING", "LOST"]:
                # Compute exact 3-sigma ellipse from eigenvalues of positional covariance Sigma_xy
                Sigma_xy = track.motion["Sigma"][:2, :2]
                eigvals, eigvecs = np.linalg.eigh(Sigma_xy)
                major_axis = float(3.0 * np.sqrt(max(float(eigvals[1]), 1e-6)))
                minor_axis = float(3.0 * np.sqrt(max(float(eigvals[0]), 1e-6)))
                angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))

                outputs.append({
                    "track_id": track.cattle_id,
                    "track_instance_id": track.track_instance_id,
                    "cattle_id": track.cattle_id,
                    "bbox": bbox,
                    "status": "occluded_virtual",
                    "sigma": Sigma_xy.tolist(),
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "angle": angle,
                    "frames_occluded": track.occlusion.get("frames_occluded", 0),
                    "velocity": vel
                })
                
        return outputs
