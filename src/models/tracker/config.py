class TrackerConfig:
    def __init__(self):
        # Clamped Dynamic Cost Weighting parameters
        self.dynamic_weight_tau = 35.0
        self.base_weights = {
            "motion": 0.35,
            "appearance": 0.25,
            "social": 0.05,
            "counterfactual": 0.15,
            "prior": 0.20
        }
        self.max_weights = {
            "appearance": 0.45,
            "prior": 0.45
        }
        
        # Memory Decay parameter
        self.memory_decay_tau = 150.0
        
        # Post-Association Re-ID parameters
        self.reid_buffer_length = 3
        self.reid_merge_threshold = 0.90
        self.reid_tentative_threshold = 0.70
        self.reid_min_average_score = 0.75
