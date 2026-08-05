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

    def extract_connected_component(self, cx, cy, occ_mask):
        """
        Extracts connected vegetation component containing (cx, cy).
        """
        if occ_mask is None:
            return None
            
        h_mask, w_mask = occ_mask.shape[:2]
        scale_x = w_mask / float(self.grid_w)
        scale_y = h_mask / float(self.grid_h)
        
        cx_s = int(np.clip(cx * scale_x, 0, w_mask - 1))
        cy_s = int(np.clip(cy * scale_y, 0, h_mask - 1))
        
        bin_mask = (occ_mask > 0).astype(np.uint8)
        if bin_mask[cy_s, cx_s] == 0:
            y_indices, x_indices = np.where(bin_mask > 0)
            if len(x_indices) > 0:
                dists = np.hypot(x_indices - cx_s, y_indices - cy_s)
                min_idx = np.argmin(dists)
                if dists[min_idx] <= 50:
                    cx_s, cy_s = x_indices[min_idx], y_indices[min_idx]
                else:
                    return None
            else:
                return None
                
        import cv2
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        target_label = labels[cy_s, cx_s]
        if target_label == 0:
            return None
            
        comp_mask = (labels == target_label)
        ys, xs = np.where(comp_mask)
        
        return {
            "mask": comp_mask,
            "min_x": float(np.min(xs) / scale_x),
            "max_x": float(np.max(xs) / scale_x),
            "min_y": float(np.min(ys) / scale_y),
            "max_y": float(np.max(ys) / scale_y),
            "center_x": float(centroids[target_label][0] / scale_x),
            "center_y": float(centroids[target_label][1] / scale_y),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "w_mask": w_mask,
            "h_mask": h_mask
        }

    def is_point_inside_component(self, cx, cy, comp_data, margin=20.0):
        if comp_data is None:
            return True
        return (comp_data["min_x"] - margin) <= cx <= (comp_data["max_x"] + margin) and \
               (comp_data["min_y"] - margin) <= cy <= (comp_data["max_y"] + margin)

    def project_to_nearest_component_point(self, cx, cy, comp_data):
        if comp_data is None:
            return cx, cy
        px = np.clip(cx, comp_data["min_x"] - 5, comp_data["max_x"] + 5)
        py = np.clip(cy, comp_data["min_y"] - 5, comp_data["max_y"] + 5)
        return float(px), float(py)
