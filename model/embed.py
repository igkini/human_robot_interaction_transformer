
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SinusoidalEmbeddingLayer(nn.Module):
    """Sinusoidal Positional Embedding for xyz and time."""

    def __init__(self, min_freq=4, max_freq=256, hidden_size=10):
        super().__init__()
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.hidden_size = hidden_size
        
        if hidden_size % 2 != 0:
            raise ValueError(f'hidden_size ({hidden_size}) must be divisible by 2.')
        
        self.num_freqs_int32 = hidden_size // 2
        self.num_freqs = float(self.num_freqs_int32)
        
        log_freq_increment = (
            math.log(float(self.max_freq) / float(self.min_freq)) /
            max(1.0, self.num_freqs - 1)
        )
        inv_freqs = self.min_freq * torch.exp(
            torch.arange(self.num_freqs, dtype=torch.float32) * -log_freq_increment
        )
        self.register_buffer('inv_freqs', inv_freqs)
    
    def forward(self, input_tensor):

        """Example:
                Input: [B, A, T, F] --> [B, A, T, F, 1] --> [B, A, T, F, hidden/2]
                Output: [B, A, T, F, hidden]
        """

        input_tensor = input_tensor.unsqueeze(-1).repeat_interleave(self.num_freqs_int32, dim=-1)
        embedded = torch.cat([
            torch.sin(input_tensor * self.inv_freqs),
            torch.cos(input_tensor * self.inv_freqs)
        ], dim=-1)
        
        return embedded

class AgentTemporalEncoder(nn.Module):
  """Encodes agents temporal positions."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key

    self.embedding_layer = SinusoidalEmbeddingLayer(
        max_freq=params.num_steps,
        hidden_size=params.feature_embedding_size)

    self.mlp = nn.Linear(
            in_features=params.feature_embedding_size,
            out_features=output_shape,
            bias=True
        )

  def _get_temporal_embedding(self, input_batch):
    
    b, a, t, _ = input_batch[self.key].shape

    t = torch.arange(0, t, dtype=torch.float32, device=input_batch[self.key].device)
    t = t[None, None, :]  # Add batch and agent dimensions
    t = t.repeat(b, a, 1)  # Expand to [b, num_agents, num_steps]
    # print(t.unsqueeze(-1).shape)

    return self.embedding_layer(t.unsqueeze(-1)) #[b, num_agents, num_steps, 1, feature_embedding_size]
    
  def forward(self, input_batch):
        
        temporal_embedding = self._get_temporal_embedding(input_batch)
        # print(temporal_embedding.shape)
        mlp_output = self.mlp(temporal_embedding)
        return mlp_output

class Agent2DOrientationEncoder(nn.Module):
    """Encodes agents 2d orientation. Input should be given in radians"""
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        self.embedding_layer = SinusoidalEmbeddingLayer(
            max_freq=2,
            hidden_size=params.feature_embedding_size//2)
        
        self.mlp = nn.Linear(
            in_features=params.feature_embedding_size, 
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        orientation = input_batch[self.key]
        orientation_embedding = torch.cat([
            self.embedding_layer(torch.sin(orientation)),
            self.embedding_layer(torch.cos(orientation))
        ], dim=-1)
        
        not_is_hidden = torch.logical_not(input_batch['is_hidden'])
        mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        return self.mlp(orientation_embedding), mask

class AgentPositionEncoder(nn.Module):
    """Encodes agents spatial positions."""

    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key

        self.sin_emb_layer = SinusoidalEmbeddingLayer(
            hidden_size= params.feature_embedding_size)
        
        self.mask_emb_layer=nn.Linear(in_features=1,
                            out_features=params.feature_embedding_size,
                            bias=True)

        self.mlp = nn.Linear(
            in_features= params.feature_embedding_size,
            out_features=output_shape,
            bias=True
        )

        self.dropout=nn.Dropout(params.drop_prob)
        self.lnorm=nn.LayerNorm(output_shape)

    def forward(self, input_batch):
        
        pos_emb = self.sin_emb_layer(input_batch[self.key])
        pos_emb=self.mlp(pos_emb)
        mask_emb=self.mask_emb_layer(input_batch[f"{self.key}/mask"]).unsqueeze(-2)

        pos_emb=pos_emb+mask_emb
        pos_emb=self.dropout(pos_emb)
        pos_emb=self.lnorm(pos_emb)
        pos_emb = self.mlp(pos_emb)

        return pos_emb
    
class AgentScalarEncoder(nn.Module):
    """Encodes a agent's scalar."""
    
    def __init__(self, key, output_shape):
        super().__init__()
        self.key = key
        
        # Input size depends on the scalar feature dimension
        self.mlp = nn.Linear(
            in_features=1,  # assuming scalar input
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        not_is_hidden = torch.logical_not(input_batch['is_hidden'])
        mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        # Add extra dimension at second-to-last position: [..., newaxis, :]
        mlp_output = self.mlp(input_batch[self.key])
        mlp_output = mlp_output.unsqueeze(-2)  # equivalent to [..., tf.newaxis, :]
        
        return mlp_output, mask

class AgentOneHotEncoder(nn.Module):
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        self.depth = params.depth
        
        self.mlp = nn.Linear(
            in_features=self.depth,
            out_features=output_shape,
            bias=True
        )

    def forward(self, input_batch):
        # Create one-hot encoding
        stage_one_hot = F.one_hot(
            input_batch[self.key].squeeze(-1).long(), 
            num_classes=self.depth
        ).float()

        # not_is_hidden = input_batch['missing_data']
        # mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        mlp_output = self.mlp(stage_one_hot)

        mlp_output = mlp_output.unsqueeze(-2) 
        return mlp_output

class AgentKeypointsEncoder(nn.Module):
    """Encodes the agent's keypoints."""
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        

        self.ff_layer1 =nn.Linear(
            in_features=params.keypoints_dim if hasattr(params, 'keypoints_dim') else 3,
            out_features=output_shape, # Change this to be smaller
            )
        
        self.kpt_attn=nn.MultiheadAttention(
            embed_dim=output_shape, 
            num_heads=params.num_heads, 
            batch_first=True,
            dropout=params.drop_prob)
        
        self.attn_ln=nn.LayerNorm(output_shape)
        

        self.ff_layer2 =nn.Linear(in_features=25*output_shape, out_features=output_shape)
        self.ff_layer3 = nn.Linear(in_features=output_shape, out_features=params.feature_embedding_size)
        
        self.ff_ln=nn.LayerNorm(output_shape)
        self.dropout=nn.Dropout(params.drop_prob)
    
    def forward(self, input_batch):
        
        # input['keypoints']: (b,a,t,k,3)
        keypoints = input_batch[self.key]
        kpt_emb = self.ff_layer1(keypoints) #(b,a,t,k,output_shape)

        b,a,t,k,f= kpt_emb.shape

        kpt_emb = kpt_emb.reshape(b*a*t,k,f)
        kpt_mask = (input_batch[f"{self.key}/mask"]).squeeze(-1)
        kpt_mask=kpt_mask.view(b*a*t,k)

        attn_out, _=self.kpt_attn(
            query=kpt_emb,
            key=kpt_emb,
            value=kpt_emb,
            key_padding_mask=kpt_mask)


        attn_out=self.attn_ln(kpt_emb + attn_out)
        attn_out=attn_out.reshape(b,a,t,k*f)
        
        kpt_emb=self.ff_layer2(attn_out)
        kpt_emb=F.relu(kpt_emb)
        kpt_emb=self.ff_layer3(kpt_emb)
        kpt_emb=self.dropout(kpt_emb)
        # kpt_emb=self.ff_ln(kpt_emb+attn_out)

        return kpt_emb.unsqueeze(-2)

class AgentHeadOrientationEncoder(nn.Module):
    """Encodes the detection stage."""
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        
        input_features = params.head_orientation_dim if hasattr(params, 'head_orientation_dim') else 3
        
        self.mlp = nn.Linear(
            in_features=input_features,
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        not_is_hidden = torch.logical_not(input_batch['is_hidden'])
        mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        mlp_output = self.mlp(input_batch[self.key])
        mlp_output = mlp_output.unsqueeze(-2)
        
        return mlp_output, mask