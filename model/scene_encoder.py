import torch
import torch.nn as nn
import torch.nn.functional as F
from .embed import SinusoidalEmbeddingLayer

class ConvOccupancyGridEncoderLayer(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.current_step_idx = params.num_history_steps
        self.num_filters = params.num_conv_filters
        self.hidden_size = params.hidden_size
        drop_prob = params.drop_prob

        self.ff_layer1 = nn.Linear(
            in_features=256*256,
            out_features=self.hidden_size,
            bias=True
        )
        self.ff_layer2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=256*256,
            bias=True
        )
        self.ff_dropout=nn.Dropout(params.drop_prob)
        self.ff_ln=nn.LayerNorm(256*256)

        layers = []
        in_channels = 3
        
        for i, num_filter in enumerate(self.num_filters):
            if i == 0 or i == 1:
                strides = 2
            else:
                strides = 1
            
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=3,
                stride=strides,
                padding=1
            )
    
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filter))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(params.drop_prob))
            pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.append(pooling_layer)
            
            in_channels = num_filter
        
        # Flatten
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prob))
        layers.append(nn.LayerNorm(self.hidden_size))
        
        self.seq_layers = nn.Sequential(*layers)
    
    def forward(self, input_batch):
        input_batch = input_batch.copy()
        
        occ_grid = input_batch['scene/grid']
        coord_grid= input_batch['scene/coord']
        
        # b,c,w,h=coord_grid.shape

        # coord_grid= coord_grid.reshape(b, c, w*h)
        # grid_enc=self.ff_layer1(coord_grid)
        # grid_enc=F.relu(grid_enc)
        # grid_enc=self.ff_layer2(grid_enc)
        # grid_enc=self.ff_dropout(grid_enc)
        # grid_enc=self.ff_ln(grid_enc+coord_grid)
        # grid_enc=grid_enc.reshape(b,c,w,h)
        
        grid=torch.cat([occ_grid,coord_grid],dim=1)
        
        # Apply convolutional layers to grid
        occ_grid = self.seq_layers(grid)
        
        out = occ_grid
        
        return out