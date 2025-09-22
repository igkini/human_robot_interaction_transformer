from .embed import AgentTemporalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class FeatureConcatAgentEncoderLayer(nn.Module):
   
    """
    MLP that connects all features
    """
    
    def __init__(self, agent_type, params):
        super().__init__()
        
        self.agent_type=agent_type
        agents_feature_config= getattr(params, f"{self.agent_type}_agents_feature_config")
        num_encoders = len(agents_feature_config) + 2
        input_size = num_encoders * (params.hidden_size)
        
        self.ff_layer = nn.Linear(
            in_features=input_size,
            out_features=params.hidden_size,
            bias=True
        )

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Feature Embeddings
        for key, layer in agents_feature_config.items():
            self.agent_feature_embedding_layers.append(
                layer(key, params.hidden_size, params)
            )
        
        # Temporal Embedding
        self.agent_feature_embedding_layers.append(
            AgentTemporalEncoder(
                list(agents_feature_config.keys())[0],
                params.hidden_size,
                params,
            )
        )
    
    def forward(self, input_batch: Dict[str, torch.Tensor]):
        
        layer_embeddings = []
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch) 
            
            # flatten last two dimensions
            original_shape = layer_embedding.shape
            new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)
            layer_embedding = layer_embedding.reshape(new_shape)

            layer_embeddings.append(layer_embedding)

        embedding = torch.cat(layer_embeddings, dim=-1)
        # Apply final feedforward layer
        out = self.ff_layer(embedding)
        
        return out