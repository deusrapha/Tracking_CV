import numpy as np

class TemporalMemory:
    """
    Manages the temporal history buffer for a CounterfactualAmodalTrack.
    Stores past positions, velocities, and tracking states.
    Supports query routing for counterfactual checks and hindsight trajectory reconciliation.
    """
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.trajectory_history = []  # list of dict: {"cx": cx, "cy": cy, "state": state, "frame": frame}
        self.velocity_history = []    # list of np.ndarray or list: [vx, vy]
        self.remerge_candidates = []  # list of candidate match records
        
    def update(self, cx, cy, state, frame, velocity=None):
        """
        Appends the current step's state to the memory buffers.
        """
        self.trajectory_history.append({
            "cx": float(cx),
            "cy": float(cy),
            "state": state,
            "frame": frame
        })
        if len(self.trajectory_history) > self.max_size:
            self.trajectory_history.pop(0)
            
        if velocity is not None:
            self.velocity_history.append(np.array(velocity, dtype=np.float32))
            if len(self.velocity_history) > self.max_size:
                self.velocity_history.pop(0)
                
    def get_recent_velocity(self, window=5):
        """
        Calculates the average velocity over the last N frames.
        """
        if not self.velocity_history:
            return np.array([0.0, 0.0], dtype=np.float32)
        recent = self.velocity_history[-window:]
        return np.mean(recent, axis=0)

    def record_remerge_candidate(self, frame, det_idx, cost, sim, dist):
        """
        Records a candidate re-emergence match for hindsight validation.
        """
        self.remerge_candidates.append({
            "frame": frame,
            "det_idx": det_idx,
            "cost": float(cost),
            "sim": float(sim),
            "dist": float(dist)
        })
        if len(self.remerge_candidates) > 50:
            self.remerge_candidates.pop(0)
            
    def clear_candidates(self):
        self.remerge_candidates.clear()
