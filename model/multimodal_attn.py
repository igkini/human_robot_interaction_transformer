import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttnTransformerLayer(nn.Module):
    """Performs full self-attention across the agent and time dimensions."""
    
    def __init__(
        self,
        num_heads=8,
        hidden_size=256,
        drop_prob=0.1,
        ln_eps=1e-6,
        ff_dim=128,
        mask_style=None,
        flatten=False,
        multimodality_induced=False,
        name=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mask_style = mask_style
        self.flatten = flatten
        self.multimodality_induced = multimodality_induced
        self.name = name if name is not None else self.__class__.__name__
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({num_heads}).'
            )
        
        # MultiHeadAttention layer
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,  # We handle dropout separately
            batch_first=True
        )
        
        self.attn_dropout = nn.Dropout(drop_prob)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
        
        # Feed-forward layers using Linear layers with einsum-like behavior
        self.ff_layer1 = nn.Linear(hidden_size, ff_dim)
        self.ff_layer2 = nn.Linear(ff_dim, hidden_size)
        
        self.ff_dropout = nn.Dropout(drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
    
    def forward(self, input_batch):
        # Create a copy of input_batch to avoid modifying the original
        input_batch = {k: v for k, v in input_batch.items()}
        
        # [b, a, t, h] or [b, a, t, n, h]
        hidden_vecs = input_batch['hidden_vecs']
        original_shape = hidden_vecs.shape
        
        if self.flatten:
            b = hidden_vecs.shape[0]
            h = hidden_vecs.shape[-1]
            
            if self.multimodality_induced:
                n = hidden_vecs.shape[3]
                # Reshape to [b, a*t, n, h]
                hidden_vecs = hidden_vecs.reshape(b, -1, n, h)
            else:
                # Reshape to [b, a*t, h]
                hidden_vecs = hidden_vecs.reshape(b, -1, h)
        
        # Handle attention mask
        attn_mask = None
        if self.mask_style is None:
            attn_mask = None
        elif self.mask_style == 'has_historic_data':
            # Shape: [b, a, t]
            has_historic_data = input_batch['has_historic_data'][..., 0]
            
            if self.flatten:
                a = original_shape[1]
                t = original_shape[2]
                
                if self.multimodality_induced:
                    # Create mask for flattened multimodal case
                    # Original mask: [b, a, t]
                    # Need: [b, a*t, a*t]
                    attn_mask = has_historic_data.unsqueeze(1).unsqueeze(2)  # [b, 1, 1, a, t]
                    attn_mask = attn_mask.unsqueeze(3).expand(-1, -1, a, t, -1, t)  # [b, 1, a, t, a, t]
                    attn_mask = attn_mask.reshape(b, a*t, a*t)
                else:
                    # Create mask for flattened case
                    # Expand to [b, a, t, a, t]
                    attn_mask = has_historic_data.unsqueeze(1).unsqueeze(3)  # [b, 1, a, 1, t]
                    attn_mask = attn_mask.expand(-1, a, -1, t, -1)  # [b, a, a, t, t]
                    # Tile and reshape
                    attn_mask = attn_mask.permute(0, 1, 3, 2, 4).reshape(b, a*t, a*t)
            else:
                # For non-flattened case, we need to handle multi-dimensional attention
                # PyTorch MultiheadAttention expects 2D attention mask
                # We'll need custom attention implementation for this case
                pass
        else:
            raise ValueError(f'Unrecognized mask style: {self.mask_style}. '
                           "Must be either None, 'has_historic_data'.")
        
        # Apply attention
        if self.flatten or (not self.flatten and hidden_vecs.dim() == 3):
            # Standard case for flattened or [b, seq, h] tensors
            # Convert mask from True=attend to True=ignore for PyTorch
            if attn_mask is not None:
                attn_mask = ~attn_mask  # Invert mask for PyTorch convention
            
            # For multihead attention, we need to handle the sequence dimension properly
            if self.multimodality_induced and self.flatten:
                # Reshape to merge modes into hidden dimension for attention
                b, seq_len, n, h = hidden_vecs.shape
                hidden_vecs_attn = hidden_vecs.reshape(b, seq_len, n * h)
                
                # Apply attention
                attn_out, attn_weights = self.attn_layer(
                    hidden_vecs_attn, 
                    hidden_vecs_attn, 
                    hidden_vecs_attn,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True
                )
                
                # Reshape back
                attn_out = attn_out.reshape(b, seq_len, n, h)
            else:
                attn_out, attn_weights = self.attn_layer(
                    hidden_vecs, 
                    hidden_vecs, 
                    hidden_vecs,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True
                )
        else:
            # For non-flattened multi-dimensional attention
            # We need to implement custom attention logic
            # This is a simplified version - you may need to adjust based on your needs
            b, a, t, h = hidden_vecs.shape
            hidden_vecs_reshaped = hidden_vecs.reshape(b, a * t, h)
            
            if attn_mask is not None:
                # Reshape mask accordingly
                attn_mask = ~attn_mask.reshape(b, a * t, a * t)
            
            attn_out, attn_weights = self.attn_layer(
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                attn_mask=attn_mask,
                need_weights=True
            )
            attn_out = attn_out.reshape(b, a, t, h)
            attn_weights = attn_weights.reshape(b, -1, a, t, a, t)  # Approximate reshape
        
        # Apply dropout and layer norm
        out = self.attn_dropout(attn_out)
        attn_out = self.attn_ln(out + hidden_vecs)
        
        # Feed-forward layers
        # Apply FF layers while preserving tensor shape
        
        out = F.relu(self.ff_layer1(attn_out))
        out = self.ff_layer2(out)
        
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        # Reshape back to original shape if flattened
        if self.flatten:
            out = out.reshape(original_shape)
        
        # Update output
        input_batch['hidden_vecs'] = out
        input_batch[f'attn_scores/{self.name}'] = attn_weights
        
        return input_batch


# Example usage:
if __name__ == "__main__":
    # Create layer
    layer = SelfAttnTransformerLayer(
        num_heads=8,
        hidden_size=256,
        drop_prob=0.1,
        ln_eps=1e-6,
        ff_dim=128,
        mask_style=None,
        flatten=True,
        multimodality_induced=False,
        name='transformer_layer_1'  # Optional: specify a custom name
    )
    
    # Create dummy input
    batch_size = 2
    num_agents = 4
    time_steps = 10
    hidden_size = 256
    
    input_batch = {
        'hidden_vecs': torch.randn(batch_size, num_agents, time_steps, hidden_size),
        'has_historic_data': torch.ones(batch_size, num_agents, time_steps, 1)
    }
    
    # Forward pass
    output_batch = layer(input_batch)
    print(f"Output shape: {output_batch['hidden_vecs'].shape}")
    print(f"Attention scores shape: {output_batch[f'attn_scores/{layer.name}'].shape}")


class SelfAttnModeTransformerLayer(nn.Module):
    """Performs full self-attention across the future modes dimensions."""
    
    def __init__(
        self,
        num_heads=8,
        hidden_size=256,
        drop_prob=0.1,
        ln_eps=1e-6,
        ff_dim=128,
        name=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.name = name if name is not None else self.__class__.__name__
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({num_heads}).'
            )
        
        # MultiHeadAttention layer
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,  # We handle dropout separately
            batch_first=True
        )
        
        self.attn_dropout = nn.Dropout(drop_prob)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
        
        # Feed-forward layers
        self.ff_layer1 = nn.Linear(hidden_size, ff_dim)
        self.ff_layer2 = nn.Linear(ff_dim, hidden_size)
        
        self.ff_dropout = nn.Dropout(drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
    
    def forward(self, input_batch, training=None):
        # Create a copy of input_batch to avoid modifying the original
        input_batch = {k: v for k, v in input_batch.items()}
        
        # [b, a, t, h] or [b, a, t, n, h]
        hidden_vecs = input_batch['hidden_vecs']
        
        # Check if we have modes dimension
        if hidden_vecs.dim() == 5:
            # [b, a, t, n, h] - attention over modes (dimension 3)
            b, a, t, n, h = hidden_vecs.shape
            
            # Reshape to put modes in sequence dimension for attention
            # Merge batch, agents, and time dimensions
            hidden_vecs_reshaped = hidden_vecs.reshape(b * a * t, n, h)
            
            # Self-attention over modes
            attn_out, attn_weights = self.attn_layer(
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                need_weights=True,
                average_attn_weights=True
            )
            
            # Reshape back
            attn_out = attn_out.reshape(b, a, t, n, h)
            # attn_weights shape: [b*a*t, n, n] (already averaged)
            attn_score = attn_weights.reshape(b, a, t, n, n)
            
        else:
            # [b, a, t, h] - no modes dimension, standard attention
            b, a, t, h = hidden_vecs.shape
            
            # For standard case, we might want to do attention over time dimension
            # Reshape to [b*a, t, h]
            hidden_vecs_reshaped = hidden_vecs.reshape(b * a, t, h)
            
            # Self-attention
            attn_out, attn_weights = self.attn_layer(
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                hidden_vecs_reshaped,
                need_weights=True,
                average_attn_weights=True
            )
            
            # Reshape back
            attn_out = attn_out.reshape(b, a, t, h)
            
            # attn_weights shape: [b*a, t, t] (already averaged)
            attn_score = attn_weights.reshape(b, a, t, t)
        
        # Apply dropout and layer norm
        out = self.attn_dropout(attn_out)
        attn_out = self.attn_ln(out + hidden_vecs)
        
        # Feed-forward layers
        # Apply FF layers while preserving tensor shape
        
        out = F.relu(self.ff_layer1(attn_out))
        out = self.ff_layer2(out)
        
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        # Update output
        input_batch['hidden_vecs'] = out
        input_batch[f'attn_scores/{self.name}'] = attn_score
        
        return input_batch

class SceneNonMultimodalCrossAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        num_heads=8,
        hidden_size=256,
        drop_prob=0.0,
        ln_eps=1e-6,
        ff_dim=128,
        name=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.name = name if name is not None else self.__class__.__name__
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({num_heads}).'
            )
        
        # MultiHeadAttention layer for cross-attention
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,  # We handle dropout separately
            batch_first=True
        )
        
        self.attn_dropout = nn.Dropout(drop_prob)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
        
        # Feed-forward layers
        self.ff_layer1 = nn.Linear(hidden_size, ff_dim)
        self.ff_layer2 = nn.Linear(ff_dim, hidden_size)
        
        self.ff_dropout = nn.Dropout(drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
    
    def forward(self, input_batch, training=None):
        # Create a copy of input_batch to avoid modifying the original
        input_batch = {k: v for k, v in input_batch.items()}
        
        # [b, a, t, h]
        hidden_vecs = input_batch['hidden_vecs']
        b, a, t, h = hidden_vecs.shape
        
        # [b, N, H]
        scene_hidden_vec = input_batch['scene_hidden_vec']
        N = scene_hidden_vec.shape[1]
        
        # Expand scene_hidden_vec to match agents and time dimensions
        # [b, N, H] -> [b, 1, 1, N, H]
        scene_hidden_vec = scene_hidden_vec.unsqueeze(1).unsqueeze(2)
        
        # Tile to [b, a, t, N, H]
        scene_hidden_vec = scene_hidden_vec.expand(b, a, t, N, h)
        
        # Add extra dimension to hidden_vecs for cross-attention
        # [b, a, t, h] -> [b, a, t, 1, h]
        hidden_vecs_ext = hidden_vecs.unsqueeze(-2)
        
        # For cross-attention, we need to reshape to handle the attention properly
        # Query: agents at each timestep query the scene
        # Key/Value: scene elements
        
        # Reshape for attention
        # Query: [b*a*t, 1, h]
        query = hidden_vecs_ext.reshape(b * a * t, 1, h)
        
        # Key/Value: [b*a*t, N, h]
        key_value = scene_hidden_vec.reshape(b * a * t, N, h)
        
        # Cross-attention: each agent-timestep queries all scene elements
        attn_out, attn_weights = self.attn_layer(
            query,
            key_value,
            key_value,
            need_weights=True,
            average_attn_weights=True  # Average across heads
        )
        
        # attn_out shape: [b*a*t, 1, h]
        # attn_weights shape: [b*a*t, 1, N] (already averaged across heads)
        
        # Reshape attention output back
        attn_out = attn_out.reshape(b, a, t, 1, h)
        
        # Remove the extra dimension (similar to [..., 0, :] in TF)
        attn_out = attn_out.squeeze(-2)  # [b, a, t, h]
        
        # Reshape attention weights (already averaged)
        attn_score = attn_weights.reshape(b, a, t, 1, N)
        
        # Apply dropout and layer norm
        out = self.attn_dropout(attn_out)
        attn_out = self.attn_ln(out + hidden_vecs)
        
        out = F.relu(self.ff_layer1(attn_out))
        out = self.ff_layer2(out)
        
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        # Update output
        input_batch['hidden_vecs'] = out
        input_batch[f'attn_scores/{self.name}'] = attn_score
        
        return input_batch


class SceneMultimodalCrossAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        num_heads=8,
        hidden_size=256,
        drop_prob=0.0,
        ln_eps=1e-6,
        ff_dim=128,
        name=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.name = name if name is not None else self.__class__.__name__
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({num_heads}).'
            )
        
        # MultiHeadAttention layer for cross-attention
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,  # We handle dropout separately
            batch_first=True
        )
        
        self.attn_dropout = nn.Dropout(drop_prob)
        self.attn_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
        
        # Feed-forward layers
        self.ff_layer1 = nn.Linear(hidden_size, ff_dim)
        self.ff_layer2 = nn.Linear(ff_dim, hidden_size)
        
        self.ff_dropout = nn.Dropout(drop_prob)
        self.ff_ln = nn.LayerNorm(hidden_size, eps=ln_eps)
    
    def forward(self, input_batch, training=None):
        # Create a copy of input_batch to avoid modifying the original
        input_batch = {k: v for k, v in input_batch.items()}
        
        # [b, a, t, n, h]
        hidden_vecs = input_batch['hidden_vecs']
        b, a, t, n, h = hidden_vecs.shape
        
        # [b, N, H]
        scene_hidden_vec = input_batch['scene_hidden_vec']
        N = scene_hidden_vec.shape[1]
        
        # Expand scene_hidden_vec to match all dimensions
        # [b, N, H] -> [b, 1, 1, 1, N, H]
        scene_hidden_vec = scene_hidden_vec.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        # Tile to [b, a, t, n, N, H]
        scene_hidden_vec = scene_hidden_vec.expand(b, a, t, n, N, h)
        
        # Add extra dimension to hidden_vecs for cross-attention
        # [b, a, t, n, h] -> [b, a, t, n, 1, h]
        hidden_vecs_ext = hidden_vecs.unsqueeze(-2)
        
        # For cross-attention with attention over dimension 4 (scene elements)
        # Reshape for attention
        # Query: [b*a*t*n, 1, h] - each agent-timestep-mode queries
        query = hidden_vecs_ext.reshape(b * a * t * n, 1, h)
        
        # Key/Value: [b*a*t*n, N, h] - all scene elements
        key_value = scene_hidden_vec.reshape(b * a * t * n, N, h)
        
        # Cross-attention: each agent-timestep-mode queries all scene elements
        attn_out, attn_weights = self.attn_layer(
            query,
            key_value,
            key_value,
            need_weights=True,
            average_attn_weights=True 
        )
        
        # attn_out shape: [b*a*t*n, 1, h]
        # attn_weights shape: [b*a*t*n, 1, N] (already averaged across heads)
        
        # Reshape attention output back
        attn_out = attn_out.reshape(b, a, t, n, 1, h)
        
        # Remove the extra dimension (similar to [..., 0, :] in TF)
        attn_out = attn_out.squeeze(-2)  # [b, a, t, n, h]
        
        # Reshape attention weights
        attn_score = attn_weights.reshape(b, a, t, n, 1, N)
        
        # Apply dropout and layer norm
        out = self.attn_dropout(attn_out)
        attn_out = self.attn_ln(out + hidden_vecs)
        
    
        out = F.relu(self.ff_layer1(attn_out))
        out = self.ff_layer2(out)
        
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        # Update output
        input_batch['hidden_vecs'] = out
        input_batch[f'attn_scores/{self.name}'] = attn_score
        
        return input_batch