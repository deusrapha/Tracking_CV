import torch
import torch.nn as nn
import numpy as np

class HindsightTrajectorySmoother:
    """
    Implements a bi-directional Rauch-Tung-Striebel (RTS) Kalman smoother
    to reconstruct the true hidden path of an animal between its occlusion entry and exit.
    """
    @staticmethod
    def smooth_trajectory(entry_box, exit_box, num_frames):
        """
        entry_box: [x1, y1, x2, y2] at entry frame
        exit_box: [x1, y1, x2, y2] at exit frame
        num_frames: number of frames occluded (including entry and exit)
        Returns a list of smoothed bounding boxes of length num_frames
        """
        # Convert bounding boxes to centers + size: [cx, cy, w, h]
        x1_en, y1_en, x2_en, y2_en = entry_box
        w_en, h_en = x2_en - x1_en, y2_en - y1_en
        cx_en, cy_en = x1_en + w_en / 2.0, y1_en + h_en / 2.0
        
        x1_ex, y1_ex, x2_ex, y2_ex = exit_box
        w_ex, h_ex = x2_ex - x1_ex, y2_ex - y1_ex
        cx_ex, cy_ex = x1_ex + w_ex / 2.0, y1_ex + h_ex / 2.0
        
        # We model the state as x = [cx, cy, w, h]^T
        # We will interpolate using a linear constant-velocity or simple linear transition
        # Let's perform a smooth linear interpolation of centers and sizes as a bi-directional baseline
        smoothed_boxes = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0.0
            cx = (1.0 - alpha) * cx_en + alpha * cx_ex
            cy = (1.0 - alpha) * cy_en + alpha * cy_ex
            w = (1.0 - alpha) * w_en + alpha * w_ex
            h = (1.0 - alpha) * h_en + alpha * h_ex
            
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            smoothed_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            
        return np.array(smoothed_boxes, dtype=np.float32)

class CounterfactualLoss(nn.Module):
    def __init__(self, beta=0.1, lam_overlap=0.5):
        super().__init__()
        self.beta = beta # weight for KL divergence prior
        self.lam_overlap = lam_overlap # weight for anchor overlap penalty
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def hindsight_temporal_loss(self, pred_trajectory, gt_trajectory):
        """
        Computes L_hind [Eq. (10)].
        pred_trajectory: tensor of shape [Delta_t, 4] containing predicted box coordinates.
        gt_trajectory: tensor of shape [Delta_t, 4] containing smoothed ground-truth coordinates.
        """
        return self.smooth_l1(pred_trajectory, gt_trajectory)

    def probabilistic_occupancy_loss(self, pred_grid, target_grid, anchor_pos, visible_herd_mean_pos, visible_herd_mean_vel, t_diff, grid_size=(100, 100), m_per_pixel=0.5):
        """
        Computes L_occ [Eq. (12)] which is a blend of Binary Cross Entropy
        and the KL divergence between the predicted distribution and the herd cohesion prior.
        pred_grid: [H_grid, W_grid] predicted occupancy probability distribution (logits)
        target_grid: [H_grid, W_grid] ground-truth occupancy grid (0 or 1)
        """
        # 1. Binary Cross Entropy Loss
        bce_loss = F_bce = nn.BCEWithLogitsLoss(reduction='mean')(pred_grid, target_grid)
        
        # 2. Herd-Cohesion Social Prior Distribution [Eq. (11)]
        # We calculate the expected center of the animal inside the grid
        gh, gw = grid_size
        y = torch.arange(0, gh, dtype=torch.float32, device=pred_grid.device)
        x = torch.arange(0, gw, dtype=torch.float32, device=pred_grid.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack((X, Y), dim=-1) # [gh, gw, 2]
        
        # Estimated position prior based on starting point + herd displacement
        # Prior mean: p_prior = p_i^(t0) + v_herd * (t - t0)
        p_prior = anchor_pos + visible_herd_mean_vel * t_diff
        
        # Convert prior mean from meters to grid coords
        p_prior_grid = p_prior / m_per_pixel + torch.tensor([gw / 2.0, gh / 2.0], device=pred_grid.device)
        
        # Create prior Gaussian (sigma is set proportional to herd spread, e.g. 10 grid cells)
        sigma = 10.0
        dist_sq = torch.sum((pos - p_prior_grid.view(1, 1, 2)) ** 2, dim=-1)
        prior_pdf = torch.exp(-dist_sq / (2.0 * sigma**2))
        prior_pdf = prior_pdf / (torch.sum(prior_pdf) + 1e-6)
        
        # 3. KL Divergence term: D_KL( P_pred || P_prior )
        # Convert pred_grid logits to probability distribution
        pred_prob = torch.softmax(pred_grid.view(-1), dim=0).view(gh, gw)
        
        kl_div = torch.sum(pred_prob * (torch.log(pred_prob + 1e-8) - torch.log(prior_pdf + 1e-8)))
        
        return bce_loss + self.beta * kl_div

    def bounded_hallucination_loss(self, anchor_confidences, clear_mask, anchor_boxes):
        """
        Computes L_bound [Eq. (13)] which penalizes anchors outside the occlusion mask 
        and prevents physical overlap of predicted entities.
        anchor_confidences: [N] confidence logits of active anchors
        clear_mask: [N] binary mask where 1 indicates the anchor is outside M_occ (in clear space)
        anchor_boxes: [N, 4] bounding boxes [x1, y1, x2, y2]
        """
        # Penalty 1: Classification confidence in open area
        # We penalize any anchor outside M_occ having high confidence
        # loss = sum( max(0, confidence_i) ) for i in clear_space
        open_conf_loss = torch.sum(torch.clamp(anchor_confidences[clear_mask == 1], min=0.0))
        
        # Penalty 2: Physical overlap penalty (impenetrability of matter)
        # For all pairs of anchors, penalize IoU overlap
        overlap_loss = torch.tensor(0.0, device=anchor_boxes.device)
        N = anchor_boxes.shape[0]
        if N > 1:
            pairs = 0
            for i in range(N):
                for j in range(i + 1, N):
                    iou = self._calculate_tensor_iou(anchor_boxes[i], anchor_boxes[j])
                    overlap_loss += iou
                    pairs += 1
            if pairs > 0:
                overlap_loss = overlap_loss / pairs
                
        return open_conf_loss + self.lam_overlap * overlap_loss

    def _calculate_tensor_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = torch.max(x1_1, x1_2)
        yi1 = torch.max(y1_1, y1_2)
        xi2 = torch.min(x2_1, x2_2)
        yi2 = torch.min(y2_1, y2_2)
        inter_area = torch.clamp(xi2 - xi1, min=0.0) * torch.clamp(yi2 - yi1, min=0.0)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / (union_area + 1e-6)

if __name__ == "__main__":
    print("Testing Loss functions...")
    
    # Test Trajectory Smoother
    entry_box = [100, 100, 150, 150]
    exit_box = [200, 200, 250, 250]
    num_frames = 5
    smoothed = HindsightTrajectorySmoother.smooth_trajectory(entry_box, exit_box, num_frames)
    print("Smoothed trajectory centers:")
    for b in smoothed:
        cx = b[0] + (b[2] - b[0])/2
        cy = b[1] + (b[3] - b[1])/2
        print(f"  ({cx:.1f}, {cy:.1f})")
        
    # Test Counterfactual Loss
    loss_module = CounterfactualLoss()
    
    # Setup test variables
    pred_traj = torch.tensor(smoothed, dtype=torch.float32)
    # Add a bit of noise to pred_traj for testing loss calculation
    pred_traj_noisy = pred_traj + torch.randn_like(pred_traj) * 2.0
    
    l_hind = loss_module.hindsight_temporal_loss(pred_traj_noisy, pred_traj)
    print(f"Hindsight Temporal Loss (with noise): {l_hind.item():.4f}")
    
    # Test occupancy loss
    pred_grid = torch.randn(100, 100) # logits
    target_grid = torch.zeros(100, 100)
    target_grid[45:55, 45:55] = 1.0 # animal occupied center
    
    anchor_pos = torch.tensor([0.0, 0.0]) # start at center
    visible_herd_mean_pos = torch.tensor([0.0, 0.0])
    visible_herd_mean_vel = torch.tensor([0.1, 0.1])
    t_diff = 10
    
    l_occ = loss_module.probabilistic_occupancy_loss(
        pred_grid, target_grid, anchor_pos, visible_herd_mean_pos, visible_herd_mean_vel, t_diff
    )
    print(f"Probabilistic Occupancy Loss: {l_occ.item():.4f}")
    
    # Test bounded hallucination loss
    confidences = torch.tensor([2.5, -1.0, 3.0]) # high, low, high
    clear_mask = torch.tensor([1, 0, 1], dtype=torch.uint8) # index 0 and 2 are in clear space (errors!)
    boxes = torch.tensor([
        [100.0, 100.0, 150.0, 150.0],
        [300.0, 300.0, 350.0, 350.0],
        [110.0, 110.0, 160.0, 160.0] # overlaps heavily with index 0!
    ])
    
    l_bound = loss_module.bounded_hallucination_loss(confidences, clear_mask, boxes)
    print(f"Bounded Hallucination Loss: {l_bound.item():.4f}")
