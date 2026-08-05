import time
import numpy as np
import torch
from baseline_tracker import BoundingBoxTracker
from projection import GroundPlaneProjector
from amodal_anchor import OcclusionAwareTracker
from attention import SpatiotemporalDeformableAttention

def run_profile_benchmark(num_frames=100, img_w=1920, img_h=1080):
    print("==================================================")
    print("Occlusion-Aware Multi-Animal Tracking Profile Benchmark")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Initialize modules
    print("Initializing framework modules...")
    projector = GroundPlaneProjector(img_w=img_w, img_h=img_h)
    baseline = BoundingBoxTracker(max_lost_frames=30)
    tracker = OcclusionAwareTracker(baseline, projector, grid_w=img_w, grid_h=img_h)
    
    # Initialize Deformable Attention Layer
    attn_layer = SpatiotemporalDeformableAttention(embed_dim=128, num_heads=4, num_frames=3, num_points=4).to(device)
    
    # 2. Prepare mock inputs
    # Let's create a simulated frame (1920x1080 RGB) with some green vegetation areas
    frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    frame[300:500, 400:600, 1] = 200 # Acacia canopy 1
    frame[600:800, 1200:1400, 1] = 210 # Acacia canopy 2
    
    # Simulate a herd of 15 cattle moving across the pasture
    # Each cattle is represented by [x1, y1, x2, y2]
    np.random.seed(42)
    cattle_positions = []
    for i in range(15):
        cx = np.random.uniform(100, img_w - 100)
        cy = np.random.uniform(100, img_h - 100)
        w, h = 60.0, 60.0
        cattle_positions.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
    cattle_positions = np.array(cattle_positions)
    # Simulated velocity vectors
    velocities = np.random.uniform(-5.0, 5.0, size=(15, 2))
    
    timings = {
        "segmentation": [],
        "homography_projection": [],
        "baseline_tracking": [],
        "deformable_attention": [],
        "total_step": []
    }
    
    # 3. Profiling Loop
    print(f"Running profile loop over {num_frames} frames with {len(cattle_positions)} targets...")
    for f in range(num_frames):
        step_start = time.perf_counter()
        
        # A. Vegetation Segmentation
        t0 = time.perf_counter()
        occ_mask = projector.segment_vegetation(frame)
        t1 = time.perf_counter()
        timings["segmentation"].append((t1 - t0) * 1000.0) # in ms
        
        # B. Ground-Plane Projection (warping mask)
        t0 = time.perf_counter()
        ground_mask = projector.project_mask_to_ground(occ_mask, altitude=30.0, pitch_deg=-60.0, roll_deg=0.0)
        t1 = time.perf_counter()
        timings["homography_projection"].append((t1 - t0) * 1000.0)
        
        # C. Update animal positions (simulate movement)
        detections = []
        for i in range(len(cattle_positions)):
            # Update position
            cattle_positions[i, 0] += velocities[i, 0]
            cattle_positions[i, 1] += velocities[i, 1]
            cattle_positions[i, 2] += velocities[i, 0]
            cattle_positions[i, 3] += velocities[i, 1]
            
            # Check if this animal falls inside a vegetation canopy (occlusion)
            cx = int((cattle_positions[i, 0] + cattle_positions[i, 2]) / 2)
            cy = int((cattle_positions[i, 1] + cattle_positions[i, 3]) / 2)
            
            # Bound check
            cx = min(max(cx, 0), img_w - 1)
            cy = min(max(cy, 0), img_h - 1)
            
            is_occluded = occ_mask[cy, cx] > 0
            if not is_occluded:
                # Add to detections if not occluded
                detections.append(list(cattle_positions[i]))
                
        # D. Baseline Tracking & Amodal Seeding
        t0 = time.perf_counter()
        active_tracks = tracker.step(frame, detections, altitude=30.0, pitch=-60.0, roll=0.0)
        t1 = time.perf_counter()
        timings["baseline_tracking"].append((t1 - t0) * 1000.0)
        
        # E. Deformable Attention feature querying
        # Only query attention if there are active amodal anchors
        num_anchors = len(tracker.amodal_anchors)
        if num_anchors > 0:
            t0 = time.perf_counter()
            # Fake query features and reference points
            query_tensor = torch.randn(num_anchors, 128, device=device)
            ref_pts = torch.randn(num_anchors, 2, device=device)
            mem_feats = torch.randn(3, 128, 32, 32, device=device)
            
            # Forward pass
            updated_query = attn_layer(query_tensor, ref_pts, mem_feats)
            t1 = time.perf_counter()
            timings["deformable_attention"].append((t1 - t0) * 1000.0)
        else:
            timings["deformable_attention"].append(0.0)
            
        step_end = time.perf_counter()
        timings["total_step"].append((step_end - step_start) * 1000.0)

    # 4. Compile and print results
    print("\n================ BENCHMARK REPORT ================")
    print(f"Resolution: {img_w}x{img_h} | Targets: {len(cattle_positions)} | Device: {device}")
    print("--------------------------------------------------")
    print(f"{'Module / Step':<32} | {'Mean Latency (ms)':<18} | {'Max Latency (ms)':<16}")
    print("--------------------------------------------------")
    for key, vals in timings.items():
        mean_val = np.mean(vals)
        max_val = np.max(vals)
        print(f"{key.replace('_', ' ').title():<32} | {mean_val:18.2f} | {max_val:16.2f}")
        
    print("--------------------------------------------------")
    total_mean = np.mean(timings["total_step"])
    est_fps = 1000.0 / total_mean if total_mean > 0 else 0
    print(f"Total Mean Frame Latency: {total_mean:.2f} ms")
    print(f"Estimated Throughput:     {est_fps:.2f} FPS")
    print("==================================================")

if __name__ == "__main__":
    run_profile_benchmark()
