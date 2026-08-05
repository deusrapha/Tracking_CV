import cv2
import numpy as np

class GroundPlaneProjector:
    def __init__(self, intrinsic_matrix=None, img_w=1920, img_h=1080):
        self.img_w = img_w
        self.img_h = img_h
        
        # Set default camera intrinsics if none provided (assuming standard DJI lens)
        if intrinsic_matrix is None:
            # Standard DJI 35mm equivalent focal length on a 1/2.3" or 1" sensor
            f = 1500.0 # focal length in pixels (approximation)
            cx = img_w / 2.0
            cy = img_h / 2.0
            self.K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.K = np.array(intrinsic_matrix, dtype=np.float32)
            
        self.K_inv = np.linalg.inv(self.K)

    def segment_vegetation(self, frame):
        """
        Segments green vegetation (Acacia tree canopies, bushes) from pasture ground
        using a combination of the Excess Green Index (ExG = 2G - R - B) and HSV green range masking.
        This provides high robustness against shadow variations and lighting changes.
        """
        # 1. Compute Excess Green Index (ExG)
        b_u8, g_u8, r_u8 = cv2.split(frame)
        b = b_u8.astype(np.float32)
        g = g_u8.astype(np.float32)
        r = r_u8.astype(np.float32)
        exg = 2.0 * g - r - b
        _, exg_thresh = cv2.threshold(exg, 30.0, 255, cv2.THRESH_BINARY)
        exg_thresh = exg_thresh.astype(np.uint8)
        
        # 2. Compute HSV Green Mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green range in HSV space: Hue [35, 85], Saturation [40, 255], Value [30, 255]
        lower_green = np.array([35, 40, 30], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        hsv_thresh = cv2.inRange(hsv, lower_green, upper_green)
        
        # 3. Fuse masks
        fused = cv2.bitwise_or(exg_thresh, hsv_thresh)
        
        # Apply morphological opening to remove small noise and closing to fill canopy holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)
        
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel)
        
        return mask

    def compute_homography(self, altitude, pitch_deg, roll_deg):
        """
        Computes the homography matrix H that maps image coordinates [u, v, 1]
        to ground coordinates [x, y, 1] on the plane z=0.
        altitude: height of the UAV in meters (z_uav)
        pitch_deg: gimbal/camera pitch angle in degrees (negative looking down, e.g. -90 for nadir)
        roll_deg: camera roll angle in degrees
        """
        # Convert angles to radians
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        
        # Rotation matrices
        # Rx (pitch) and Ry (roll)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ], dtype=np.float32)
        
        R_roll = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)]
        ], dtype=np.float32)
        
        # Combined rotation matrix (R = R_pitch @ R_roll)
        R = R_pitch @ R_roll
        
        # Translation vector (UAV is at [0, 0, altitude])
        t = np.array([0, 0, altitude], dtype=np.float32).reshape(3, 1)
        
        # The plane normal in camera space is n_c = R^T * [0, 0, 1]^T
        # But since z=0 in ground coordinates:
        # P_ground = [X, Y, 0]^T
        # We can map: P_camera = R * P_ground + t
        # P_camera = R_1 * X + R_2 * Y + t
        # P_camera = [R_1, R_2, t] * [X, Y, 1]^T
        # And since image coordinates are p = K * P_camera / Z_c,
        # we have p ~ K * [R_1, R_2, t] * [X, Y, 1]^T
        # Therefore, the ground-to-image homography is:
        # H_g2i = K @ [R_1, R_2, t] (where R_1, R_2 are first two columns of R)
        # We need the image-to-ground homography, which is the inverse:
        # H_i2g = H_g2i^-1
        
        R_1 = R[:, 0].reshape(3, 1)
        R_2 = R[:, 1].reshape(3, 1)
        
        H_g2i = self.K @ np.hstack((R_1, R_2, t))
        
        # Check if matrix is singular
        if np.linalg.cond(H_g2i) < 1/sys.float_info.epsilon if 'sys' in globals() else True:
            H_i2g = np.linalg.inv(H_g2i)
        else:
            # Fallback to identity
            H_i2g = np.eye(3, dtype=np.float32)
            
        return H_i2g

    def project_mask_to_ground(self, mask, altitude, pitch_deg, roll_deg, grid_size=(100, 100), m_per_pixel=0.5):
        """
        Projects the 2D image mask to a ground-plane coordinate grid.
        Returns a binary grid representation of the occlusion mask M_occ.
        grid_size: size of the output ground occupancy grid (height, width)
        m_per_pixel: scale of the ground grid in meters per grid cell
        """
        H_i2g = self.compute_homography(altitude, pitch_deg, roll_deg)
        
        # Apply perspective warp to the image mask to project it onto the ground plane
        # Define output size and offset to keep coordinates centered
        gh, gw = grid_size
        
        # Create destination mapping: ground center is at (gw/2, gh/2)
        # Translate from ground meters to grid coordinates
        T_ground_to_grid = np.array([
            [1.0 / m_per_pixel, 0, gw / 2.0],
            [0, 1.0 / m_per_pixel, gh / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Full warp matrix from image to ground grid
        M_warp = T_ground_to_grid @ H_i2g
        
        # Warp the mask
        ground_mask = cv2.warpPerspective(
            mask,
            M_warp,
            (gw, gh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return ground_mask

if __name__ == "__main__":
    import sys
    # Quick test code
    print("Testing GroundPlaneProjector...")
    projector = GroundPlaneProjector()
    
    # Create fake green frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Fill a region with green
    frame[400:600, 800:1100, 1] = 200 # Green channel high
    
    mask = projector.segment_vegetation(frame)
    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
    
    # Project to ground at altitude 30m, pitch -60 (oblique view), roll 0
    ground_mask = projector.project_mask_to_ground(mask, altitude=30.0, pitch_deg=-60.0, roll_deg=0.0)
    print(f"Ground mask shape: {ground_mask.shape}, sum of values: {np.sum(ground_mask > 0)}")
