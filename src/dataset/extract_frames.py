import os
import zipfile
import cv2
import glob
import shutil
import json

# Paths
WORKSPACE_DIR = r"c:\New folder\Local Disk\Masters MCS\SEM_II\Tracking_CV"
DATASET_DIR = os.path.join(WORKSPACE_DIR, "Dataset")
TEMP_DIR = os.path.join(DATASET_DIR, "temp_extracted")
OUT_DIR = os.path.join(DATASET_DIR, "processed_frames")

TARGET_FPS = 5.0

def process_video(video_path, out_video_dir):
    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if native_fps <= 0:
        native_fps = 25.0 # fallback
        
    step = max(1, int(round(native_fps / TARGET_FPS)))
    
    print(f"  Processing {os.path.basename(video_path)}: {width}x{height} @ {native_fps:.2f} FPS ({total_frames} frames).")
    print(f"  Downsampling to {TARGET_FPS} FPS by extracting every {step}-th frame...")
    
    frame_idx = 0
    saved_count = 0
    mapping = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            saved_count += 1
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(out_video_dir, frame_name)
            
            # Save frame
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # Map saved frame number to original frame index and timestamp
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            mapping[saved_count] = {
                "original_frame_index": frame_idx,
                "timestamp_ms": timestamp_ms
            }
            
        frame_idx += 1
        
    cap.release()
    
    # Save mapping and metadata
    metadata = {
        "video_name": os.path.basename(video_path),
        "width": width,
        "height": height,
        "native_fps": native_fps,
        "target_fps": TARGET_FPS,
        "total_original_frames": total_frames,
        "extracted_frames": saved_count,
        "frame_mapping": mapping
    }
    
    with open(os.path.join(out_video_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"  Successfully extracted {saved_count} frames to {out_video_dir}.")
    return True

def main():
    print("==================================================")
    # Ensure directories exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    # Find all zip files
    zip_files = sorted(glob.glob(os.path.join(DATASET_DIR, "videos-*.zip")))
    print(f"Found {len(zip_files)} zip archive files.")
    
    for zf_path in zip_files:
        print(f"\nProcessing zip archive: {os.path.basename(zf_path)}")
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                namelist = zf.namelist()
                # Find all MP4 files inside the zip
                mp4_files = [x for x in namelist if x.lower().endswith('.mp4')]
                print(f"Found {len(mp4_files)} MP4 videos in zip.")
                
                for member in mp4_files:
                    video_name = os.path.basename(member)
                    video_id = os.path.splitext(video_name)[0]
                    out_video_dir = os.path.join(OUT_DIR, video_id)
                    
                    # Skip if already processed (check if metadata.json exists)
                    if os.path.exists(os.path.join(out_video_dir, "metadata.json")):
                        print(f"  Video {video_name} already processed. Skipping.")
                        continue
                        
                    print(f"  Extracting {member} to temp folder...")
                    extracted_path = zf.extract(member, TEMP_DIR)
                    
                    # Process extracted video
                    success = process_video(extracted_path, out_video_dir)
                    
                    # Clean up temp file
                    try:
                        os.remove(extracted_path)
                        print(f"  Cleaned up temp video {video_name}.")
                    except Exception as e:
                        print(f"  Warning: Could not delete temp video {extracted_path}: {e}")
                        
        except Exception as e:
            print(f"Error reading zip file {zf_path}: {e}")
            
    # Remove temp directory if empty
    try:
        shutil.rmtree(TEMP_DIR)
        print("\nCleaned up temp directory.")
    except Exception as e:
        print(f"\nCould not delete temp directory {TEMP_DIR}: {e}")
        
    print("==================================================")
    print("Frame extraction pipeline completed.")

if __name__ == "__main__":
    main()
