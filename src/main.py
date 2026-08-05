import os
import json
import glob
import cv2
import numpy as np
import time
import gc
from PIL import Image
import subprocess
import argparse

from models.projection import GroundPlaneProjector
from models.tracker import CounterfactualAmodalTracker
from models.evaluation import MOTEvaluator

# Paths
WORKSPACE_DIR = r"c:\New folder\Local Disk\Masters MCS\SEM_II\Tracking_CV"
DATASET_DIR = os.path.join(WORKSPACE_DIR, "Dataset")
FRAMES_DIR = os.path.join(DATASET_DIR, "processed_frames")

class HSVAppearanceExtractor:
    def __init__(self):
        self.mode = "hsv"
        
    def extract_features_batch(self, frame, bboxes):
        return [self.compute_hsv_hist(frame, bbox) for bbox in bboxes]
        
    def extract_features(self, frame, bbox):
        return self.compute_hsv_hist(frame, bbox)
        
    def compute_hsv_hist(self, frame, bbox):
        h, w, _ = frame.shape
        # Scale bounding box from 1920x1080 grid to actual frame size
        scale_x = w / 1920.0
        scale_y = h / 1080.0
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(64, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

def simulate_yolo_detections(metadata_path, noise_level=2.0):
    """
    Simulates a YOLO detector by loading the ground truth annotations
    and adding slight coordinate noise and random false negatives/positives
    to create realistic detector outputs.
    """
    if not os.path.exists(metadata_path):
        return None
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    # We will simulate detections based on the frames listed in metadata.json
    # Since we are running on unannotated videos for dry-run validation, we will
    # generate synthetic target trajectories that move across the pasture and get occluded
    # by segmented vegetation masks.
    return meta

def run_pipeline(target_video_id=None, extractor="hsv"):
    print("==================================================")
    print("Beyond the Line of Sight - End-to-End Pipeline")
    print("==================================================")
    
    # Find all processed video directories
    video_dirs = sorted([d for d in glob.glob(os.path.join(FRAMES_DIR, "*")) if os.path.isdir(d)])
    print(f"Found {len(video_dirs)} processed video directories.")
    
    if target_video_id is None:
        global_results = {}
        for v_dir in video_dirs:
            video_id = os.path.basename(v_dir)
            metadata_path = os.path.join(v_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
                
            print(f"\n[COORDINATOR] Launching subprocess for Video: {video_id}")
            cmd = ["python", "src/main.py", "--video", video_id, "--extractor", extractor]
            subprocess.run(cmd, check=True)
            
            # Load the generated metrics report from the subprocess
            metrics_path = os.path.join(v_dir, "metrics_report.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_report = json.load(f)
                global_results[video_id] = {
                    "fps": 0.0,
                    "metrics": metrics_report
                }
                
        # Write aggregated pipeline summary
        summary_path = os.path.join(FRAMES_DIR, "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(global_results, f, indent=4)
            
        print("\n==================================================")
        print(f"Pipeline executed successfully. Global summary: {summary_path}")
        print("==================================================")
        return
        
    # Initialize global models (inside isolated subprocess)
    projector = GroundPlaneProjector()
    evaluator = MOTEvaluator()
    if extractor == "hsv":
        shared_extractor = HSVAppearanceExtractor()
    else:
        from models.baseline_tracker import AppearanceExtractor
        shared_extractor = AppearanceExtractor(onnx_path="src/models/reid_model.onnx")
        shared_extractor.mode = extractor
    
    global_results = {}
    
    for v_dir in video_dirs:
        video_id = os.path.basename(v_dir)
        if video_id != target_video_id:
            continue
            
        metadata_path = os.path.join(v_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"Skipping {video_id} (no metadata.json found).")
            continue
            
        print(f"\nProcessing Video: {video_id}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        frame_paths = sorted(glob.glob(os.path.join(v_dir, "frame_*.jpg")))
        print(f"  Loaded {len(frame_paths)} frames. native FPS: {meta['native_fps']:.2f}")
        
        # Initialize tracker for this specific video
        tracker = CounterfactualAmodalTracker(projector=projector, extractor=shared_extractor)
        
        # Structure to save outputs
        pred_tracks = {}
        
        # For evaluation, we will simulate a ground-truth trajectory and a noisy detection set
        # to verify the quantitative metric calculation on the real video frames.
        # Let's define a mock GT trajectory for 2 cows grazing.
        gt_tracks = {os.path.basename(p): [] for p in frame_paths}
        
        # Mock cattle positions that move into segmented vegetation masks
        np.random.seed(42)
        cows = [
            {"id": 1, "cx": 300.0, "cy": 300.0, "vx": 2.0, "vy": 0.5, "w": 50, "h": 50},
            {"id": 2, "cx": 800.0, "cy": 500.0, "vx": -1.5, "vy": 1.0, "w": 55, "h": 55}
        ]
        
        t_start = time.perf_counter()
        
        for frame_idx, f_path in enumerate(frame_paths):
            frame_name = os.path.basename(f_path)
            # Read frame and downsample immediately to save 75% memory
            try:
                with Image.open(f_path) as pil_img:
                    pil_img_resized = pil_img.resize((960, 540), Image.NEAREST)
                    frame = cv2.cvtColor(np.array(pil_img_resized), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
            
            occ_mask = projector.segment_vegetation(frame)
            
            # Update mock GT and detections
            detections = []
            for cow in cows:
                # Update GT location
                cow["cx"] += cow["vx"] + np.random.normal(0, 0.2)
                cow["cy"] += cow["vy"] + np.random.normal(0, 0.2)
                
                # Check bounds on 1920x1080 scale
                cow["cx"] = min(max(cow["cx"], 0), 1919)
                cow["cy"] = min(max(cow["cy"], 0), 1079)
                
                # Bbox coords
                x1 = cow["cx"] - cow["w"]/2
                y1 = cow["cy"] - cow["h"]/2
                x2 = cow["cx"] + cow["w"]/2
                y2 = cow["cy"] + cow["h"]/2
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                
                # Check if centroid is in segmented vegetation mask (occlusion), scaling coordinates
                mask_h, mask_w = occ_mask.shape[:2]
                scale_x = mask_w / 1920.0
                scale_y = mask_h / 1080.0
                iy = min(max(int(cow["cy"] * scale_y), 0), mask_h - 1)
                ix = min(max(int(cow["cx"] * scale_x), 0), mask_w - 1)
                is_occ = occ_mask[iy, ix] > 0
                
                gt_tracks[frame_name].append({
                    "track_id": cow["id"],
                    "bbox": bbox,
                    "is_occluded": is_occ
                })
                
                if not is_occ:
                    # Detector only sees visible cows, adding slight coordinate noise (pixel dev=1.5)
                    noise_x = np.random.normal(0, 1.5)
                    noise_y = np.random.normal(0, 1.5)
                    detections.append([x1 + noise_x, y1 + noise_y, x2 + noise_x, y2 + noise_y])
            
            # Run tracker step (verbose only for targeted diagnostic frames in DJI_0001)
            verbose_step = (video_id == "DJI_0001" and frame_idx >= 1705 and frame_idx <= 1725)
            outputs = tracker.step(frame, detections, altitude=30.0, pitch=-90.0, roll=0.0, verbose=verbose_step)
            pred_tracks[frame_name] = outputs
            
            # Print detailed Counterfactual Amodal prediction diagnostics during occlusion
            if verbose_step:
                for track in tracker.tracks:
                    if track.state in ["OCCLUDED", "REMERGING"]:
                        matching_cow = next((c for c in cows if c["id"] == track.track_id), None)
                        if matching_cow is not None:
                            gt_cx = matching_cow["cx"]
                            gt_cy = matching_cow["cy"]
                            gt_vx = matching_cow["vx"]
                            gt_vy = matching_cow["vy"]
                            gt_v = np.hypot(gt_vx, gt_vy)
                            gt_theta = np.arctan2(gt_vy, gt_vx)
                            
                            pred_cx = float(track.motion["x"][0, 0])
                            pred_cy = float(track.motion["x"][1, 0])
                            pred_v = float(track.motion["x"][2, 0])
                            pred_theta = float(track.motion["x"][3, 0])
                            pred_omega = float(track.motion["x"][4, 0])
                            
                            pred_err = np.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
                            std_x = np.sqrt(track.motion["Sigma"][0, 0])
                            std_y = np.sqrt(track.motion["Sigma"][1, 1])
                            max_dev = 3.5 * max(std_x, std_y)
                            cov_trace = float(track.motion["P"][0, 0] + track.motion["P"][1, 1])
                            
                            print(
                                f"[AMODAL DIAGNOSTIC] Frame {frame_idx} | TrackID {track.track_id} | State {track.state} | "
                                f"GT ({gt_cx:.1f}, {gt_cy:.1f}) | Pred ({pred_cx:.1f}, {pred_cy:.1f}) | Error {pred_err:.1f}px | "
                                f"Heading GT {gt_theta:.3f} | Heading Pred {pred_theta:.3f} | "
                                f"Vel GT {gt_v:.2f} | Vel Pred {pred_v:.2f} | Omega {pred_omega:.4f} | "
                                f"Confidence {track.counterfactual_confidence:.3f} | CovTrace {cov_trace:.1f} | Ellipse {max_dev:.1f}"
                            )
            del frame
            del occ_mask
            
            if frame_idx % 10 == 0:
                gc.collect()
            
        t_end = time.perf_counter()
        elapsed_sec = t_end - t_start
        fps = len(frame_paths) / elapsed_sec if elapsed_sec > 0 else 0
        print(f"  Processed in {elapsed_sec:.2f} seconds ({fps:.2f} FPS).")
        
        # Save predicted tracks JSON
        pred_json_path = os.path.join(v_dir, "predicted_tracks.json")
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(pred_tracks, f, indent=4)
            
        # Export predicted tracks to MOT text formats
        vis_mot = []
        amodal_mot = []
        for f_idx, f_path in enumerate(frame_paths, start=1):
            f_name = os.path.basename(f_path)
            for track in pred_tracks.get(f_name, []):
                tid = track["track_id"]
                bbox = track["bbox"]
                left = bbox[0]
                top = bbox[1]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                if track["status"] == "visible":
                    vis_mot.append(f"{f_idx},{tid},{left:.2f},{top:.2f},{w:.2f},{h:.2f},1,-1,-1\n")
                amodal_mot.append(f"{f_idx},{tid},{left:.2f},{top:.2f},{w:.2f},{h:.2f},1,-1,-1\n")
                
        with open(os.path.join(v_dir, "pred_visible.txt"), "w", encoding="utf-8") as f:
            f.writelines(vis_mot)
        with open(os.path.join(v_dir, "pred_amodal.txt"), "w", encoding="utf-8") as f:
            f.writelines(amodal_mot)
            
        # Run quantitative evaluation against ground-truth for this sequence
        metrics_report = evaluator.evaluate_sequence(gt_tracks, pred_tracks)
        print("  Quantitative Metrics:")
        print(f"    MOTA Accuracy:          {metrics_report['mota']*100:.2f}%")
        print(f"    Identity Switches:      {metrics_report['identity_switches']}")
        print(f"    Visible Recall:         {metrics_report['visible_stratified_recall']*100:.2f}%")
        print(f"    Occluded Recall:        {metrics_report['occluded_stratified_recall']*100:.2f}%")
        print(f"    Occlusion Recovery Rate: {metrics_report['recovery_success_rate']*100:.2f}% ({metrics_report['successful_recoveries']}/{metrics_report['occlusion_events_recorded']})")
        
        # Save local metrics report
        with open(os.path.join(v_dir, "metrics_report.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=4)
            
        global_results[video_id] = {
            "fps": fps,
            "metrics": metrics_report
        }
        
        # Clean up memory to avoid allocation errors
        del tracker
        del pred_tracks
        del gt_tracks
        gc.collect()
        
    # Write aggregated pipeline summary
    summary_path = os.path.join(FRAMES_DIR, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=4)
        
    print("\n==================================================")
    print(f"Pipeline executed successfully. Global summary: {summary_path}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Process a specific video ID in an isolated subprocess")
    parser.add_argument("--extractor", type=str, default="hsv", choices=["hsv", "onnx", "trt", "pytorch"], help="Appearance feature extractor mode")
    args = parser.parse_args()
    run_pipeline(target_video_id=args.video, extractor=args.extractor)
