import numpy as np

class CanopyOcclusionHandler:
    """
    Handles canopy occlusion queries by checking if track positions lie
    within the segmented vegetation canopy mask.
    """
    def __init__(self, grid_w=1920, grid_h=1080):
        self.grid_w = grid_w
        self.grid_h = grid_h
        
    def is_occluded(self, cx, cy, occ_mask, w_box=50.0, h_box=50.0):
        """
        Checks if the given centroid coordinate (cx, cy) or its immediate neighborhood
        is within the canopy occlusion mask, scaling coordinates if the mask is resized.
        """
        if occ_mask is None:
            return False
            
        h_mask, w_mask = occ_mask.shape[:2]
        
        # Calculate scaling factors based on mask dimensions relative to default grid size
        scale_x = w_mask / float(self.grid_w)
        scale_y = h_mask / float(self.grid_h)
        
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        w_scaled = w_box * scale_x
        h_scaled = h_box * scale_y
        
        # Check a neighborhood region (15% of box size) around centroid
        rx = max(2, int(w_scaled * 0.15))
        ry = max(2, int(h_scaled * 0.15))
        
        ix1 = min(max(int(cx_scaled - rx), 0), w_mask - 1)
        ix2 = min(max(int(cx_scaled + rx), 0), w_mask - 1)
        iy1 = min(max(int(cy_scaled - ry), 0), h_mask - 1)
        iy2 = min(max(int(cy_scaled + ry), 0), h_mask - 1)
        
        # If any pixel in the neighborhood is canopy, return True
        sub_mask = occ_mask[iy1:iy2+1, ix1:ix2+1]
        return bool(np.any(sub_mask > 0))
