import os
import sys
import time
import json
import numpy as np
import cv2

# Add simulation/src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.baseline_tracker import BoundingBoxTracker, AppearanceExtractor
from models.projection import GroundPlaneProjector
from models.amodal_anchor import OcclusionAwareTracker

class HILSafetyWatchdog:
    """
    Simulates on-board watchdogs and registers safety faults (GPS loss, geofence, low confidence).
    """
    def __init__(self, geofence_poly):
        self.geofence_poly = np.array(geofence_poly, dtype=np.float32)
        self.gps_signal = 1.0 # 1.0 = good, 0.0 = lost
        self.link_signal = 1.0
        self.logs = []
        
    def check_geofence(self, uav_pos):
        # Ray-casting geofence check
        x, y = uav_pos
        dist = cv2.pointPolygonTest(self.geofence_poly, (x, y), False)
        if dist < 0:
            msg = f"[WATCHDOG FAULT] Geofence breached at pos: {uav_pos}! Triggering immediate abort."
            self.logs.append({"timestamp": time.time(), "event": "GEOFENCE_BREACH", "details": msg})
            return False, "ABORT_GEOFENCE"
        return True, "OK"
        
    def check_signals(self, gps_status, link_status):
        self.gps_signal = gps_status
        self.link_signal = link_status
        if self.gps_signal <= 0.0 or self.link_signal <= 0.0:
            msg = f"[WATCHDOG FAULT] Link or GPS signal lost! (GPS: {self.gps_signal}, Link: {self.link_signal}). Triggering RTL (Return To Land)."
            self.logs.append({"timestamp": time.time(), "event": "SIGNAL_LOSS", "details": msg})
            return False, "TRIGGER_RTL"
        return True, "OK"
        
    def check_confidence(self, mean_score):
        if mean_score < 0.25:
            msg = f"[WATCHDOG WARNING] Tracking confidence critically low ({mean_score:.2f})! Requesting manual operator takeover."
            self.logs.append({"timestamp": time.time(), "event": "LOW_CONFIDENCE", "details": msg})
            return False, "OPERATOR_TAKEOVER"
        return True, "OK"

def run_hil_simulation():
    print("==================================================")
    print("Desktop / HIL Tracking Simulation and Dry-Run")
    print("==================================================")
    
    # 1. Setup paths and test data (use DJI_0004 since it is fast and verified)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, "Dataset", "processed_frames", "DJI_0004")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Calibration sequence folder '{dataset_dir}' not found.")
        return
        
    frame_paths = sorted(glob_file_paths(dataset_dir, "*.jpg"))
    if not frame_paths:
        print("Error: No frames found in DJI_0004.")
        return
        
    print(f"Loaded {len(frame_paths)} frames for simulation replay.")
    
    # Initialize trackers and watchdogs
    # Geofence boundary: 1000m x 1000m square centered at 500,500
    geofence = [[100, 100], [900, 100], [900, 900], [100, 900]]
    watchdog = HILSafetyWatchdog(geofence)
    
    shared_extractor = AppearanceExtractor()
    projector = GroundPlaneProjector()
    baseline = BoundingBoxTracker(iou_threshold=0.3, max_lost_frames=30, extractor=shared_extractor)
    tracker = OcclusionAwareTracker(baseline, projector)
    
    # Latency tracking buffers
    e2e_latencies = []
    step_breakdowns = {"load": [], "segment": [], "cmc": [], "tracking": [], "watchdog": []}
    
    # 2. Replay loop with dynamic fault injection
    print("\nStarting HIL operations simulation run...")
    for frame_idx, f_path in enumerate(frame_paths):
        t_start = time.perf_counter()
        
        # A. Mock GStreamer filesrc capture / frame load
        t0 = time.perf_counter()
        frame = cv2.imread(f_path)
        if frame is None:
            continue
        step_breakdowns["load"].append((time.perf_counter() - t0) * 1000.0)
        
        # B. Fused canopy segmentation
        t1 = time.perf_counter()
        occ_mask = projector.segment_vegetation(frame)
        step_breakdowns["segment"].append((time.perf_counter() - t1) * 1000.0)
        
        # C. Telemetry and camera motion compensation (CMC)
        # Synthesize IMU gyro yaw, pitch, roll, and UAV position
        t2 = time.perf_counter()
        uav_pos = [500.0 + frame_idx * 5.0, 500.0 - frame_idx * 2.0] # moving within geofence
        yaw, pitch, roll = 0.05 * frame_idx, -0.02, 0.01
        
        # Fault Injection: Frame 6 has an artificial yaw spike (sudden camera swing)
        if frame_idx == 6:
            print("[FAULT INJECT] Injecting artificial camera yaw spike (yaw: 15.5 deg).")
            yaw = 15.5
            
        # Warp state using camera motion compensation (CMC)
        tracker.baseline_tracker.apply_camera_motion_compensation(frame)
        step_breakdowns["cmc"].append((time.perf_counter() - t2) * 1000.0)
        
        # D. Target detection simulation and tracking updates
        t3 = time.perf_counter()
        # Mock detection boxes representing cows moving
        detections = [[200.0 + frame_idx * 1.5, 200.0 + frame_idx * 0.8, 250.0 + frame_idx * 1.5, 250.0 + frame_idx * 0.8]]
        
        # Fault Injection: Frame 12 has a low-confidence drop
        det_score = 0.92
        if frame_idx == 12:
            print("[FAULT INJECT] Injecting low-confidence target detection score (score: 0.15).")
            det_score = 0.15
            
        tracker.step(frame, detections, altitude=30.0, pitch=-90.0, roll=0.0)
        step_breakdowns["tracking"].append((time.perf_counter() - t3) * 1000.0)
        
        # E. Watchdog safety evaluations
        t4 = time.perf_counter()
        gps_status = 1.0
        link_status = 1.0
        
        # Fault Injection: Frame 4 has GPS signal drop
        if frame_idx == 4:
            print("[FAULT INJECT] Injecting GPS signal loss (strength: 0.0).")
            gps_status = 0.0
            
        # Fault Injection: Frame 8 has a Geofence breach
        if frame_idx == 8:
            print("[FAULT INJECT] Injecting UAV out-of-bounds coordinates (pos: [950.0, 950.0]).")
            uav_pos = [950.0, 950.0]
            
        # Run safety assessments
        sig_ok, sig_action = watchdog.check_signals(gps_status, link_status)
        geo_ok, geo_action = watchdog.check_geofence(uav_pos)
        conf_ok, conf_action = watchdog.check_confidence(det_score)
        
        if not sig_ok:
            print(f"  Watchdog alarm triggered: {sig_action} ({watchdog.logs[-1]['details']})")
        if not geo_ok:
            print(f"  Watchdog alarm triggered: {geo_action} ({watchdog.logs[-1]['details']})")
        if not conf_ok:
            print(f"  Watchdog alarm triggered: {conf_action} ({watchdog.logs[-1]['details']})")
            
        step_breakdowns["watchdog"].append((time.perf_counter() - t4) * 1000.0)
        
        # Full Step E2E Latency
        step_time = (time.perf_counter() - t_start) * 1000.0
        e2e_latencies.append(step_time)
        
    # 3. Compile HIL Simulation Performance Report
    print("\nSimulation complete. Processing timing report...")
    p50_latency = np.percentile(e2e_latencies, 50)
    p95_latency = np.percentile(e2e_latencies, 95)
    mean_latencies = {k: np.mean(v) for k, v in step_breakdowns.items()}
    
    print(f"E2E Latency: p50 = {p50_latency:.2f} ms, p95 = {p95_latency:.2f} ms")
    print(f"Watchdog trigger log contains {len(watchdog.logs)} alarm events.")
    
    # Save simulated logs
    summary = {
        "e2e_latencies": e2e_latencies,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "mean_breakdowns": mean_latencies,
        "watchdog_logs": watchdog.logs
    }
    
    summary_path = os.path.join(script_dir, "simulation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Simulated operations results logged to '{summary_path}'")

def glob_file_paths(directory, pattern):
    import glob
    return glob.glob(os.path.join(directory, pattern))

if __name__ == "__main__":
    run_hil_simulation()
