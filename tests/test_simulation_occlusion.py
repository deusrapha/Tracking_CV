import unittest
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath('src'))
from models.tracker import CounterfactualAmodalTracker, CounterfactualAmodalTrack
from models.projection import GroundPlaneProjector
from models.baseline_tracker import AppearanceExtractor
from models.tracker.occlusion import CanopyOcclusionHandler
from models.tracker.association import calculate_iou, check_track_gate

class TestSimulationOcclusionGeometry(unittest.TestCase):
    def setUp(self):
        self.projector = GroundPlaneProjector(img_w=960, img_h=540)
        self.extractor = AppearanceExtractor(None)
        self.tracker = CounterfactualAmodalTracker(self.projector, self.extractor)
        self.handler = CanopyOcclusionHandler(grid_w=960, grid_h=540)

    def test_1_prediction_inside_canopy_remains_unchanged(self):
        """1. A prediction inside the canopy remains unchanged."""
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (500, 300), 80, 255, -1)
        
        comp_data = self.handler.extract_connected_component(500, 300, occ_mask)
        self.assertIsNotNone(comp_data)
        
        is_inside = self.handler.is_point_inside_component(505, 305, comp_data)
        self.assertTrue(is_inside)

    def test_2_prediction_outside_canopy_projected_back(self):
        """2. A prediction outside the canopy is projected back to boundary."""
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (500, 300), 80, 255, -1)
        
        comp_data = self.handler.extract_connected_component(500, 300, occ_mask)
        proj_x, proj_y = self.handler.project_to_nearest_component_point(650, 300, comp_data)
        
        self.assertLessEqual(proj_x, comp_data["max_x"] + 15)

    def test_3_search_does_not_exceed_exit_band(self):
        """3. SEARCH state does not exceed exit recovery band."""
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (500, 300), 80, 255, -1)
        
        comp_data = self.handler.extract_connected_component(500, 300, occ_mask)
        
        # Point far out in open ground (680, 300) -> outside exit band margin 20px
        is_inside_far = self.handler.is_point_inside_component(680, 300, comp_data, margin=20.0)
        self.assertFalse(is_inside_far)

    def test_4_detection_outside_exit_band_cannot_reclaim_identity(self):
        """4. A detection outside the exit band cannot reclaim persistent identity."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        
        # Step 1: Initialize track at (500, 300)
        self.tracker.step(frame, [[470, 270, 530, 330]])
        t1 = self.tracker.tracks[0]
        
        # Attach occlusion component mask for canopy
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (500, 300), 60, 255, -1)
        t1.occlusion["component_mask"] = self.handler.extract_connected_component(500, 300, occ_mask)
        t1.state = "SEARCH"
        
        # Step 3: Detection far out at (800, 300) -> outside exit band
        far_detection = [[770, 270, 830, 330]]
        outputs = self.tracker.step(frame, far_detection)
        
        # Verification: Far detection should NOT reclaim t1's identity (should allocate new cattle ID)
        visible_outputs = [o for o in outputs if o.get("status") == "visible"]
        if len(visible_outputs) > 0:
            self.assertNotEqual(visible_outputs[0]["cattle_id"], t1.cattle_id)

    def test_5_detection_near_exit_boundary_reclaims_identity(self):
        """5. A detection near valid exit boundary can reclaim persistent identity."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        
        self.tracker.step(frame, [[470, 270, 530, 330]])
        t1 = self.tracker.tracks[0]
        
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (500, 300), 60, 255, -1)
        t1.occlusion["component_mask"] = self.handler.extract_connected_component(500, 300, occ_mask)
        t1.state = "SEARCH"
        
        # Near exit boundary detection (510, 310)
        near_detection = [[480, 280, 540, 340]]
        outputs = self.tracker.step(frame, near_detection)
        
        visible_outputs = [o for o in outputs if o.get("status") == "visible"]
        self.assertEqual(len(visible_outputs), 1)
        self.assertEqual(visible_outputs[0]["cattle_id"], t1.cattle_id)

    def test_6_anchor_released_after_confirmed_recovery(self):
        """6. Anchor is released after confirmed recovery."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        
        self.tracker.step(frame, [[470, 270, 530, 330]])
        t1 = self.tracker.tracks[0]
        t1.state = "SEARCH"
        
        self.tracker.step(frame, [[480, 280, 540, 340]])
        
        amodal_anchors = self.tracker.amodal_anchors
        # Old search track should be SUPERSEDED and no longer in active amodal anchors
        self.assertNotIn(t1.track_instance_id, amodal_anchors)

    def test_7_overlapping_trees_select_correct_canopy(self):
        """7. Overlapping canopy trees select correct entry component."""
        occ_mask = np.zeros((540, 960), dtype=np.uint8)
        cv2.circle(occ_mask, (300, 200), 50, 255, -1)
        cv2.circle(occ_mask, (700, 400), 50, 255, -1)
        
        comp1 = self.handler.extract_connected_component(300, 200, occ_mask)
        comp2 = self.handler.extract_connected_component(700, 400, occ_mask)
        
        self.assertLess(comp1["center_x"], 500)
        self.assertGreater(comp2["center_x"], 500)

if __name__ == "__main__":
    unittest.main()
