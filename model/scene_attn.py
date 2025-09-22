import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneCrossAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        params
    ):
        super().__init__()
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({params.hidden_size}) must be an integer times'
                f' bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=params.hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob, 
            batch_first=True
        )
        
        self.attn_ln = nn.LayerNorm(params.hidden_size)
        
        self.ff_layer1 = nn.Linear(params.hidden_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, params.hidden_size)
        
        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(params.hidden_size)
    
    def forward(self, input_batch, scene_enc):
        
        # [b, a, t, h]
        b, a, t, h = input_batch.shape
        
        # b, H= scene_ctx.shape
        scene_enc=scene_enc.unsqueeze(1).unsqueeze(1)
        scene_enc = scene_enc.expand(b, a, t, h)
        query = input_batch.reshape(b * a, t, h)
        key_value = scene_enc.reshape(b * a, t, h)
        
        # Cross-attention: each agent-timestep queries all scene elements
        attn_out,_ = self.attn_layer(
            query=query,
            key=key_value,
            value=key_value,
            average_attn_weights=True
        )
        
        # attn_out shape: [b*a, t, h]
        
        attn_out = attn_out.reshape(b, a, t, h)
        attn_out = self.attn_ln(attn_out + input_batch)
        
        out = F.relu(self.ff_layer1(attn_out))
        out = self.ff_layer2(out)
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        return out
    