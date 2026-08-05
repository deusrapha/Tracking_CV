import os
import json
import glob
import cv2
import numpy as np

# In a full setup, you would install sam2:
# pip install git+https://github.com/facebookresearch/segment-anything-2.git
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 library is not installed in this environment.")
    print("Running sam2_annotate.py in simulation/dry-run mode.")

class SAM2CattleAnnotator:
    def __init__(self, workspace_dir=r"c:\New folder\Local Disk\Masters MCS\SEM_II\Tracking_CV"):
        self.workspace_dir = workspace_dir
        self.dataset_dir = os.path.join(workspace_dir, "Dataset")
        self.frames_dir = os.path.join(self.dataset_dir, "processed_frames")
        
        # SAM 2 configuration (paths to model checkpoint and config)
        self.sam2_checkpoint = os.path.join(workspace_dir, "models", "sam2_hiera_large.pt")
        self.model_cfg = "sam2_hiera_l.yaml"
        self.predictor = None
        
        if SAM2_AVAILABLE and os.path.exists(self.sam2_checkpoint):
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
            print("SAM 2 Video Predictor initialized successfully.")
            
    def get_video_frames(self, video_id):
        video_frame_dir = os.path.join(self.frames_dir, video_id)
        if not os.path.exists(video_frame_dir):
            raise FileNotFoundError(f"Frame directory not found for {video_id}")
            
        frame_paths = sorted(glob.glob(os.path.join(video_frame_dir, "frame_*.jpg")))
        return frame_paths, video_frame_dir

    def run_annotation_pipeline(self, video_id, initial_prompts=None, out_json_path=None):
        """
        Runs the annotation pipeline.
        initial_prompts: list of dicts, e.g.
            [
                {
                    "target_id": 1,
                    "frame_idx": 1, # 1-indexed
                    "points": [[x, y]], # coordinate points for clicks
                    "labels": [1],      # 1 for foreground, 0 for background
                    "box": [x_min, y_min, x_max, y_max] # optional initial bbox
                }
            ]
        """
        frame_paths, frame_dir = self.get_video_frames(video_id)
        print(f"Loaded {len(frame_paths)} frames from {frame_dir}.")
        
        if not out_json_path:
            out_json_path = os.path.join(frame_dir, "gt_annotations.json")
            
        # Structure to hold annotations for the entire video
        # Keys will be frame indices, values will be lists of object annotations
        annotations = {os.path.basename(p): [] for p in frame_paths}
        
        if not SAM2_AVAILABLE or not self.predictor:
            print("\n--- SIMULATION MODE (No PyTorch/SAM2 Model Loaded) ---")
            print("To run the real SAM 2 annotation, install SAM 2 and place the weights file.")
            print("Simulating box propagation and occlusion detection...")
            
            # Simulate basic annotations using a mock tracker
            self._simulate_annotations(frame_paths, initial_prompts, annotations)
        else:
            # Real SAM 2 implementation
            self._run_real_sam2(frame_dir, frame_paths, initial_prompts, annotations)
            
        # Save annotations to JSON
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=4)
        print(f"Saved {len(annotations)} frame annotations to {out_json_path}")
        
        # Export to MOT format as well
        self.export_to_mot_format(video_id, annotations, frame_paths)
        
    def _simulate_annotations(self, frame_paths, initial_prompts, annotations):
        """
        Simulate animal tracks and occlusion states based on initial prompts
        to show how the data structure works.
        """
        if not initial_prompts:
            # Seed standard mock prompts if none provided
            initial_prompts = [
                {"target_id": 1, "frame_idx": 1, "box": [500, 400, 560, 480]},
                {"target_id": 2, "frame_idx": 1, "box": [800, 600, 870, 680]}
            ]
            
        for prompt in initial_prompts:
            target_id = prompt["target_id"]
            start_frame = prompt["frame_idx"] - 1
            box = np.array(prompt["box"], dtype=float)
            
            # Simulated constant-velocity/grazing motion with occasional occlusion
            velocity = np.array([1.5, 0.5]) # pixels per frame
            
            for i in range(start_frame, len(frame_paths)):
                frame_name = os.path.basename(frame_paths[i])
                
                # Apply motion update
                box[0] += velocity[0] + np.random.normal(0, 0.5)
                box[1] += velocity[1] + np.random.normal(0, 0.5)
                box[2] += velocity[0] + np.random.normal(0, 0.5)
                box[3] += velocity[1] + np.random.normal(0, 0.5)
                
                # Simulate an occlusion zone (e.g. a rectangle at center of screen)
                # Let's say center of screen represents a tree canopy
                # [600, 300, 900, 600]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                
                is_occluded = bool((600 < cx < 900) and (300 < cy < 600))
                
                # Amodal bounding box is the true box
                amodal_box = [float(x) for x in box]
                
                if is_occluded:
                    # Visible box becomes smaller or empty
                    # Here we simulate that the visible box is completely empty (completely occluded)
                    visible_box = []
                    occlusion_ratio = 1.0
                    occluder_type = "vegetation"
                else:
                    visible_box = [float(x) for x in box]
                    occlusion_ratio = 0.0
                    occluder_type = "none"
                    
                annotations[frame_name].append({
                    "target_id": int(target_id),
                    "visible_bbox": visible_box,     # [x_min, y_min, x_max, y_max] or []
                    "amodal_bbox": amodal_box,       # [x_min, y_min, x_max, y_max]
                    "is_occluded": is_occluded,
                    "occlusion_ratio": float(occlusion_ratio),
                    "occluder_type": occluder_type
                })

    def _run_real_sam2(self, frame_dir, frame_paths, initial_prompts, annotations):
        """
        Execute actual SAM 2 segment tracking and extract amodal/visible bounding boxes.
        """
        import torch
        # 1. Initialize inference state
        inference_state = self.predictor.init_state(video_path=frame_dir)
        
        # 2. Add initial prompts (points, labels, or boxes)
        for prompt in initial_prompts:
            target_id = prompt["target_id"]
            frame_idx = prompt["frame_idx"] - 1 # SAM 2 uses 0-indexed frame indices
            
            if "box" in prompt:
                box_coords = np.array(prompt["box"], dtype=np.float32)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=target_id,
                    box=box_coords
                )
            elif "points" in prompt:
                points = np.array(prompt["points"], dtype=np.float32)
                labels = np.array(prompt["labels"], dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=target_id,
                    points=points,
                    labels=labels
                )
                
        # 3. Propagate tracking throughout the video
        print("Propagating masks through the video using SAM 2...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame_name = os.path.basename(frame_paths[out_frame_idx])
            
            for i, obj_id in enumerate(out_obj_ids):
                # Extract mask from logits
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                
                # Check if mask is empty
                if not np.any(mask):
                    # Mask is completely empty (object is fully occluded or lost)
                    visible_box = []
                    is_occluded = True
                    occlusion_ratio = 1.0
                    occluder_type = "vegetation" # assumption
                    
                    # For amodal box, we keep the last known box or let our amodal module predict
                    # SAM 2 itself doesn't predict amodal boxes when fully occluded. We'll fallback to
                    # a dummy propagation here.
                    last_known = self._get_last_known_bbox(annotations, frame_paths, out_frame_idx, obj_id)
                    amodal_box = last_known if last_known else []
                else:
                    # Compute bounding box of the visible segmentation mask
                    y_indices, x_indices = np.where(mask)
                    x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
                    y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
                    
                    visible_box = [x_min, y_min, x_max, y_max]
                    is_occluded = False
                    occlusion_ratio = 0.0
                    occluder_type = "none"
                    amodal_box = visible_box
                    
                annotations[frame_name].append({
                    "target_id": obj_id,
                    "visible_bbox": visible_box,
                    "amodal_bbox": amodal_box,
                    "is_occluded": is_occluded,
                    "occlusion_ratio": occlusion_ratio,
                    "occluder_type": occluder_type
                })
                
        # Reset predictor state
        self.predictor.reset_state(inference_state)

    def _get_last_known_bbox(self, annotations, frame_paths, current_idx, obj_id):
        # Look backwards in annotations to find the most recent non-empty visible bbox
        for idx in range(current_idx - 1, -1, -1):
            frame_name = os.path.basename(frame_paths[idx])
            for ann in annotations.get(frame_name, []):
                if ann["target_id"] == obj_id and ann["visible_bbox"]:
                    return ann["visible_bbox"]
        return None

    def export_to_mot_format(self, video_id, annotations, frame_paths):
        """
        Exports annotations to standard MOT files:
        - gt_visible.txt: standard tracking format for visible portions
        - gt_amodal.txt: standard tracking format for complete amodal estimations
        """
        video_frame_dir = os.path.join(self.frames_dir, video_id)
        
        vis_mot_path = os.path.join(video_frame_dir, "gt_visible.txt")
        amodal_mot_path = os.path.join(video_frame_dir, "gt_amodal.txt")
        
        vis_lines = []
        amodal_lines = []
        
        for frame_num, frame_path in enumerate(frame_paths, start=1):
            frame_name = os.path.basename(frame_path)
            frame_anns = annotations.get(frame_name, [])
            
            for ann in frame_anns:
                target_id = ann["target_id"]
                vis_box = ann["visible_bbox"]
                amodal_box = ann["amodal_bbox"]
                
                # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>
                if vis_box:
                    left = vis_box[0]
                    top = vis_box[1]
                    width = vis_box[2] - vis_box[0]
                    height = vis_box[3] - vis_box[1]
                    # Write visible line
                    vis_lines.append(f"{frame_num},{target_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1\n")
                    
                if amodal_box:
                    left = amodal_box[0]
                    top = amodal_box[1]
                    width = amodal_box[2] - amodal_box[0]
                    height = amodal_box[3] - amodal_box[1]
                    # Write amodal line
                    amodal_lines.append(f"{frame_num},{target_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1\n")
                    
        with open(vis_mot_path, "w", encoding="utf-8") as f:
            f.writelines(vis_lines)
            
        with open(amodal_mot_path, "w", encoding="utf-8") as f:
            f.writelines(amodal_lines)
            
        print(f"Exported visible tracks to {vis_mot_path}")
        print(f"Exported amodal tracks to {amodal_mot_path}")

if __name__ == "__main__":
    annotator = SAM2CattleAnnotator()
    # Simple dry-run test on DJI_0004
    try:
        annotator.run_annotation_pipeline("DJI_0004")
    except Exception as e:
        print(f"Pipeline error: {e}")
