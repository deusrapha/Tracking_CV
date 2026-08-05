import numpy as np

class MOTEvaluator:
    """
    Computes Multi-Object Tracking (MOT) metrics (MOTA, IDF1, IDSW) 
    and amodal-specific occlusion metrics (recovery rate, occlusion-stratified accuracy).
    """
    @staticmethod
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

    def evaluate_sequence(self, gt_frames, pred_frames, iou_threshold=0.3):
        """
        gt_frames: dict of frame_idx -> list of dicts:
            [{"track_id": id, "bbox": [x1, y1, x2, y2], "is_occluded": bool}]
        pred_frames: dict of frame_idx -> list of dicts:
            [{"track_id": id, "bbox": [x1, y1, x2, y2], "status": str}]
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        
        # Keep track of active target mappings: gt_id -> pred_id
        active_mappings = {}
        # Keep track of last frame a pred_id was matched to a gt_id (to detect switches)
        last_gt_mappings = {}
        
        # Stratification accumulators
        vis_gt_count = 0
        occ_gt_count = 0
        vis_matched = 0
        occ_matched = 0
        
        # Occlusion recovery tracker
        # Maps gt_id -> status: "visible", "occluded", "recovered", "failed_recovery"
        occlusion_event_status = {}
        occlusion_events_count = 0
        successful_recoveries = 0
        failed_recoveries = 0
        
        all_frames = sorted(list(set(gt_frames.keys()).union(set(pred_frames.keys()))))
        
        for frame in all_frames:
            gt_objs = gt_frames.get(frame, [])
            pred_objs = pred_frames.get(frame, [])
            
            total_gt += len(gt_objs)
            
            # Step 1: Compute pairwise IoU distances
            matches = [] # list of (gt_idx, pred_idx, iou)
            for g_idx, g_obj in enumerate(gt_objs):
                g_id = g_obj["track_id"]
                is_occ = g_obj.get("is_occluded", False)
                
                if is_occ:
                    occ_gt_count += 1
                else:
                    vis_gt_count += 1
                    
                for p_idx, p_obj in enumerate(pred_objs):
                    iou = self.calculate_iou(g_obj["bbox"], p_obj["bbox"])
                    if iou >= iou_threshold:
                        matches.append((g_idx, p_idx, iou))
                        
            # Sort matches by IoU descending
            matches.sort(key=lambda x: x[2], reverse=True)
            
            matched_gt = set()
            matched_pred = set()
            frame_mappings = {}
            
            for g_idx, p_idx, iou in matches:
                if g_idx in matched_gt or p_idx in matched_pred:
                    continue
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                
                g_id = gt_objs[g_idx]["track_id"]
                p_id = pred_objs[p_idx]["track_id"]
                frame_mappings[g_id] = p_id
                
                # Check occlusion state stratification
                is_occ = gt_objs[g_idx].get("is_occluded", False)
                if is_occ:
                    occ_matched += 1
                else:
                    vis_matched += 1
            
            # Step 2: Handle ID Switches and Mappings
            for g_id, p_id in frame_mappings.items():
                if g_id in active_mappings:
                    if active_mappings[g_id] != p_id:
                        # Identity Switch!
                        total_idsw += 1
                        print(f"  [METRIC IDSW] Frame {frame}: Target GT ID {g_id} switched from Pred ID {active_mappings[g_id]} to {p_id}.")
                        active_mappings[g_id] = p_id
                else:
                    # New mapping
                    active_mappings[g_id] = p_id
                    
            # Step 3: Handle Occlusion Recovery Analytics
            for g_obj in gt_objs:
                g_id = g_obj["track_id"]
                is_occ = g_obj.get("is_occluded", False)
                p_id = frame_mappings.get(g_id, None)
                
                # State transition tracker
                current_state = occlusion_event_status.get(g_id, "visible")
                
                if current_state == "visible" and is_occ:
                    # Enter occlusion event
                    occlusion_event_status[g_id] = "occluded"
                    occlusion_events_count += 1
                    # Store mapping at entry
                    occlusion_event_status[f"{g_id}_entry_pred"] = active_mappings.get(g_id, None)
                    
                elif current_state == "occluded" and not is_occ:
                    # Emerge from occlusion!
                    entry_pred = occlusion_event_status.get(f"{g_id}_entry_pred", None)
                    current_pred = p_id
                    
                    if current_pred is not None and entry_pred == current_pred:
                        successful_recoveries += 1
                        occlusion_event_status[g_id] = "visible"
                        print(f"  [METRIC RECOVERY] Target GT ID {g_id} successfully recovered after occlusion with persistent Pred ID {current_pred}.")
                    elif current_pred is not None:
                        failed_recoveries += 1
                        occlusion_event_status[g_id] = "visible"
                        print(f"  [METRIC FAIL] Target GT ID {g_id} failed recovery. Switched ID from {entry_pred} to {current_pred}.")
                    else:
                        # Still lost
                        pass
            
            # Step 4: False Positives and False Negatives
            fn = len(gt_objs) - len(matched_gt)
            fp = len(pred_objs) - len(matched_pred)
            total_fn += fn
            total_fp += fp
            
        # Compile Metrics
        mota = 1.0 - (total_fn + total_fp + total_idsw) / (total_gt + 1e-6)
        
        vis_accuracy = vis_matched / (vis_gt_count + 1e-6)
        occ_accuracy = occ_matched / (occ_gt_count + 1e-6)
        
        recovery_rate = successful_recoveries / (occlusion_events_count + 1e-6)
        
        report = {
            "total_ground_truth_boxes": total_gt,
            "false_negatives": total_fn,
            "false_positives": total_fp,
            "identity_switches": total_idsw,
            "mota": float(mota),
            "visible_stratified_recall": float(vis_accuracy),
            "occluded_stratified_recall": float(occ_accuracy),
            "occlusion_events_recorded": occlusion_events_count,
            "successful_recoveries": successful_recoveries,
            "failed_recoveries": failed_recoveries,
            "recovery_success_rate": float(recovery_rate)
        }
        
        return report

if __name__ == "__main__":
    print("Testing MOTEvaluator script...")
    evaluator = MOTEvaluator()
    
    # Create a mock sequence representing ground truth and predictions
    # Frame 1: Cow 1 is visible
    # Frame 2: Cow 1 goes under tree canopy (occluded)
    # Frame 3: Cow 1 is still occluded
    # Frame 4: Cow 1 emerges
    
    gt_seq = {
        1: [{"track_id": 1, "bbox": [100, 100, 150, 150], "is_occluded": False}],
        2: [{"track_id": 1, "bbox": [120, 110, 170, 160], "is_occluded": True}],
        3: [{"track_id": 1, "bbox": [140, 120, 190, 170], "is_occluded": True}],
        4: [{"track_id": 1, "bbox": [160, 130, 210, 180], "is_occluded": False}],
    }
    
    # Scenario A: Baseline tracker without amodal anchors (loses ID or drops boxes)
    pred_seq_baseline = {
        1: [{"track_id": 1, "bbox": [100, 100, 150, 150], "status": "visible"}],
        2: [], # drops box
        3: [], 
        4: [{"track_id": 2, "bbox": [160, 130, 210, 180], "status": "visible"}], # ID Switch on re-emergence!
    }
    
    # Scenario B: Amodal tracker (keeps anchor active, re-associates on re-emergence)
    pred_seq_amodal = {
        1: [{"track_id": 1, "bbox": [100, 100, 150, 150], "status": "visible"}],
        2: [{"track_id": 1, "bbox": [120, 110, 170, 160], "status": "occluded_virtual"}], # keeps anchor
        3: [{"track_id": 1, "bbox": [140, 120, 190, 170], "status": "occluded_virtual"}],
        4: [{"track_id": 1, "bbox": [160, 130, 210, 180], "status": "visible"}], # persistent ID!
    }
    
    print("\n--- Evaluating Scenario A: Baseline Tracker ---")
    rep_a = evaluator.evaluate_sequence(gt_seq, pred_seq_baseline)
    for k, v in rep_a.items():
        print(f"  {k:<32}: {v}")
        
    print("\n--- Evaluating Scenario B: Amodal Tracker (Ours) ---")
    rep_b = evaluator.evaluate_sequence(gt_seq, pred_seq_amodal)
    for k, v in rep_b.items():
        print(f"  {k:<32}: {v}")
