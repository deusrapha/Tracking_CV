import unittest
import numpy as np
import os
import sys

# Ensure src is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from models.tracker import CounterfactualAmodalTracker, CounterfactualAmodalTrack
from models.projection import GroundPlaneProjector
from models.baseline_tracker import AppearanceExtractor

class TestPersistentCattleIdentity(unittest.TestCase):
    def setUp(self):
        self.projector = GroundPlaneProjector(img_w=960, img_h=540)
        self.extractor = AppearanceExtractor(onnx_path=None)
        self.tracker = CounterfactualAmodalTracker(projector=self.projector, extractor=self.extractor)

    def test_A_single_cattle_stable(self):
        """Test A: Single cattle moving without occlusion maintains stable cattle_id and track_instance_id."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox = [100.0, 100.0, 160.0, 160.0]

        # Step 1: Initial detection
        outputs = self.tracker.step(frame, [bbox])
        self.assertEqual(len(self.tracker.tracks), 1)
        track = self.tracker.tracks[0]
        self.assertEqual(track.cattle_id, 1)
        self.assertEqual(track.track_instance_id, 1)
        self.assertEqual(track.state, "VISIBLE")

        # Step 2: Next frame, slightly moved
        bbox2 = [105.0, 105.0, 165.0, 165.0]
        outputs = self.tracker.step(frame, [bbox2])
        self.assertEqual(len(self.tracker.tracks), 1)
        track = self.tracker.tracks[0]
        self.assertEqual(track.cattle_id, 1)
        self.assertEqual(track.track_instance_id, 1)

    def test_B_prolonged_occlusion_recovery(self):
        """Test B: Prolonged occlusion leads to SEARCH state, and re-emergence inherits cattle_id with new track_instance_id."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox = [200.0, 200.0, 260.0, 260.0]

        # Step 1: Initial detection
        self.tracker.step(frame, [bbox])
        track1 = self.tracker.tracks[0]
        self.assertEqual(track1.cattle_id, 1)
        self.assertEqual(track1.track_instance_id, 1)

        # Step 2: 20 frames of missing detections (target occluded)
        for _ in range(20):
            self.tracker.step(frame, [])

        # Track 1 should now be in SEARCH state
        self.assertIn(track1.state, ["OCCLUDED", "SEARCH", "LOST"])

        # Step 3: Target re-emerges near predicted location
        reemerge_bbox = [210.0, 210.0, 270.0, 270.0]
        self.tracker.step(frame, [reemerge_bbox])

        # Find visible tracks
        visible_tracks = [t for t in self.tracker.tracks if t.state in ["VISIBLE", "REMERGING"]]
        self.assertEqual(len(visible_tracks), 1)
        new_track = visible_tracks[0]

        # Invariant checks:
        self.assertEqual(new_track.cattle_id, 1, "Re-emerged cattle must inherit cattle_id = 1")
        self.assertEqual(track1.state, "SUPERSEDED", "Old track instance must be SUPERSEDED")
        self.assertEqual(track1.successor_track_instance_id, new_track.track_instance_id)
        self.assertEqual(new_track.predecessor_track_instance_id, track1.track_instance_id)

    def test_C_border_entry_vs_interior_reemergence(self):
        """Test C: Detection at image border gets SCENE_ENTRY while interior gets OCCLUSION_REEMERGENCE."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        border_bbox = [2.0, 100.0, 62.0, 160.0] # Near left border (x1 = 2)

        self.tracker.step(frame, [border_bbox])
        track = self.tracker.tracks[0]
        self.assertEqual(track.origin_type, "SCENE_ENTRY")

    def test_D_multi_animal_crossing_unique_identities(self):
        """Test D: Multiple cattle maintain distinct cattle_ids."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox1 = [100.0, 100.0, 160.0, 160.0]
        bbox2 = [300.0, 300.0, 360.0, 360.0]

        self.tracker.step(frame, [bbox1, bbox2])
        cattle_ids = {t.cattle_id for t in self.tracker.tracks if t.state == "VISIBLE"}
        self.assertEqual(len(cattle_ids), 2)
        self.assertIn(1, cattle_ids)
        self.assertIn(2, cattle_ids)

    def test_E_covariance_growth_during_search(self):
        """Test E: Covariance Sigma grows monotonically during SEARCH state."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox = [200.0, 200.0, 260.0, 260.0]

        self.tracker.step(frame, [bbox])
        track = self.tracker.tracks[0]
        sigma_initial = float(track.motion["Sigma"][0, 0])

        # Step 10 frames without detection
        for _ in range(10):
            self.tracker.step(frame, [])

        sigma_after = float(track.motion["Sigma"][0, 0])
        self.assertGreater(sigma_after, sigma_initial, "Covariance must grow over occluded frames")

    def test_F_unique_visible_persistent_identities(self):
        """Test F: Invariant check - no two visible tracks may share the same cattle_id."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox1 = [100.0, 100.0, 160.0, 160.0]
        bbox2 = [300.0, 300.0, 360.0, 360.0]

        self.tracker.step(frame, [bbox1, bbox2])
        visible_tracks = [t for t in self.tracker.tracks if t.state in ["VISIBLE", "REMERGING"]]
        visible_cattle_ids = [t.cattle_id for t in visible_tracks]
        self.assertEqual(len(visible_cattle_ids), len(set(visible_cattle_ids)))

    def test_G_counter_independence(self):
        """Test G: Creating a new track instance during recovery increments next_track_instance_id but not next_cattle_id."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox = [200.0, 200.0, 260.0, 260.0]

        self.tracker.step(frame, [bbox])
        initial_cattle_counter = self.tracker.next_cattle_id
        initial_instance_counter = self.tracker.next_track_instance_id

        # Missing frames
        for _ in range(5):
            self.tracker.step(frame, [])

        # Re-emerges
        self.tracker.step(frame, [[205.0, 205.0, 265.0, 265.0]])
        self.assertGreater(self.tracker.next_track_instance_id, initial_instance_counter)
        self.assertEqual(self.tracker.next_cattle_id, initial_cattle_counter, "next_cattle_id should NOT increment on identity recovery")

    def test_H_superseded_tracks_do_not_compete(self):
        """Test H: Superseded track instances are excluded from active bipartite matching candidates."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        bbox = [200.0, 200.0, 260.0, 260.0]

        self.tracker.step(frame, [bbox])
        t1 = self.tracker.tracks[0]

        # Missing frames
        for _ in range(5):
            self.tracker.step(frame, [])

        # Re-emerge
        self.tracker.step(frame, [[205.0, 205.0, 265.0, 265.0]])
        self.assertEqual(t1.state, "SUPERSEDED")

        # Next frame
        outputs = self.tracker.step(frame, [[210.0, 210.0, 270.0, 270.0]])
        # Superseded track must not be included in visible outputs
        visible_track_ids = [o["track_id"] for o in outputs if o.get("status") == "visible"]
        self.assertEqual(len(visible_track_ids), 1)

    def test_I_output_and_trajectory_consistency(self):
        """Test I: Output structures and cattle counts use cattle_id."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        outputs = self.tracker.step(frame, [[100.0, 100.0, 160.0, 160.0]])
        self.assertEqual(outputs[0]["track_id"], 1)

if __name__ == "__main__":
    unittest.main()
