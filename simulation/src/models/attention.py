import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalDeformableAttention(nn.Module):
    """
    Implements the Spatiotemporal Deformable Attention layer [Eq. (7), (8), (9)]
    for updating Amodal Anchor query vectors by retrieving contextual features 
    from a sliding queue of past feature maps.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_frames=3, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_frames = num_frames # T (number of memory frames)
        self.num_points = num_points # L (number of sampling points per frame)
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections to predict sampling offsets
        # For each query, we predict (num_frames * num_heads * num_points * 2) offsets (x, y)
        self.sampling_offsets = nn.Linear(embed_dim, num_frames * num_heads * num_points * 2)
        
        # Linear projections to predict attention weights
        self.attention_weights = nn.Linear(embed_dim, num_frames * num_heads * num_points)
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offset projections to small values (so they start sampling near the reference point)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        
        # Initialize attention weights projection
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, memory_features):
        """
        query: tensor of shape [N, C] where N is number of queries, C is embed_dim.
        reference_points: tensor of shape [N, 2] containing (x, y) coordinates in normalized image space [-1, 1].
        memory_features: tensor of shape [T, C, H, W] representing the sliding queue of T past frame feature maps.
        """
        N, C = query.shape
        T, _, H, W = memory_features.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames in memory, got {T}"
        
        # 1. Predict offsets and attention weights from query
        # shape: [N, T * num_heads * num_points * 2]
        offsets = self.sampling_offsets(query)
        # Reshape to [N, T, num_heads, num_points, 2]
        offsets = offsets.view(N, T, self.num_heads, self.num_points, 2)
        
        # shape: [N, T * num_heads * num_points]
        attn_logits = self.attention_weights(query)
        # Reshape to [N, T * num_heads, num_points] and apply softmax
        attn_weights = F.softmax(attn_logits.view(N, T * self.num_heads, self.num_points), dim=-1)
        # Reshape back to [N, T, num_heads, num_points]
        attn_weights = attn_weights.view(N, T, self.num_heads, self.num_points)
        
        # 2. Prepare value features by projecting memory features
        # Reshape features to [T, H * W, C] for linear projection, then back to [T, C, H, W]
        # Memory features shape: [T, C, H, W] -> Transpose/reshape
        value_feats = memory_features.permute(0, 2, 3, 1).reshape(T * H * W, C)
        value_feats = self.value_proj(value_feats).view(T, H, W, C).permute(0, 3, 1, 2) # [T, C, H, W]
        
        # Split value features into heads: [T, num_heads, head_dim, H, W]
        value_feats = value_feats.view(T, self.num_heads, self.head_dim, H, W)
        
        # 3. Perform bilinear sampling over reference points + offsets
        output = torch.zeros((N, self.num_heads, self.head_dim), device=query.device, dtype=query.dtype)
        
        # Normalize offsets relative to feature map dimensions (scale down)
        # offsets are predicted in absolute pixels or normalized?
        # Here we assume offsets are predicted in normalized units of scale 0.1 of frame size
        scale = torch.tensor([0.1, 0.1], device=query.device, dtype=query.dtype)
        
        for t in range(T):
            for h in range(self.num_heads):
                # For head h and frame t, compute sample grid points
                # query reference_points: [N, 2]
                # offsets: [N, T, num_heads, num_points, 2]
                # sample_coords shape: [N, num_points, 2]
                sample_coords = reference_points.unsqueeze(1) + offsets[:, t, h, :, :] * scale.view(1, 1, 2)
                
                # Check coords are within [-1, 1] range for grid_sample
                sample_coords = torch.clamp(sample_coords, -1.0, 1.0)
                
                # grid_sample expects input shape [Batch, Channels, H_in, W_in]
                # and grid shape [Batch, H_out, W_out, 2]
                # We can treat each query as a separate batch element for sampling:
                # input: [N, head_dim, H, W]
                # grid: [N, 1, num_points, 2]
                
                # Expand head value feature map for all queries
                # value_feats[t, h] shape is [head_dim, H, W]
                grid_input = value_feats[t, h].unsqueeze(0).expand(N, -1, -1, -1) # [N, head_dim, H, W]
                grid = sample_coords.unsqueeze(1) # [N, 1, num_points, 2]
                
                # Sample features
                # sampled shape: [N, head_dim, 1, num_points]
                sampled = F.grid_sample(grid_input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                sampled = sampled.squeeze(2) # [N, head_dim, num_points]
                
                # Multiply by attention weights: attn_weights[:, t, h, :] has shape [N, num_points]
                weights = attn_weights[:, t, h, :].unsqueeze(1) # [N, 1, num_points]
                
                # Weighted sum over sampling points
                head_out = torch.sum(sampled * weights, dim=-1) # [N, head_dim]
                
                output[:, h, :] += head_out
                
        # 4. Project combined head features back to query dimension
        output = output.view(N, C)
        output = self.output_proj(output)
        
        # Residual connection
        return query + output

if __name__ == "__main__":
    print("Testing SpatiotemporalDeformableAttention module...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device:", device)
    
    # Init layer
    attn_layer = SpatiotemporalDeformableAttention(embed_dim=128, num_heads=4, num_frames=3, num_points=4).to(device)
    
    # Create fake input tensors
    # N=2 queries, C=128 channels
    query = torch.randn(2, 128, device=device)
    # reference points at center (0, 0) and bottom-right (0.5, 0.5)
    ref_pts = torch.tensor([[0.0, 0.0], [0.5, 0.5]], device=device, dtype=torch.float32)
    # T=3 frames, C=128, H=32, W=32 feature maps
    mem_feats = torch.randn(3, 128, 32, 32, device=device)
    
    # Forward pass
    out = attn_layer(query, ref_pts, mem_feats)
    print("Input query shape:", query.shape)
    print("Output query shape:", out.shape)
    print("Is output finite:", torch.all(torch.isfinite(out)).item())
