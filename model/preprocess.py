import torch
import torch.nn as nn
import torch.nn.functional as F

class GridPreprocessLayer(nn.Module):
    def __init__(self, output_size=(256, 256), coord_scale=6.425000):
        super().__init__()
        self.output_size = output_size
        self.coord_scale = coord_scale
        
        # Can add learnable parameters here if needed
        # self.norm_mean = nn.Parameter(torch.tensor([0.5]))
        # self.norm_std = nn.Parameter(torch.tensor([0.5]))
        
    def create_coordinate_grid(self, batch_size, height, width, origin, resolution, device):
        """Create coordinate grids for a batch of maps."""
        # Create pixel coordinate grids
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        
        # Create meshgrid [H, W]
        pixel_y, pixel_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand for batch dimension [B, H, W]
        pixel_x = pixel_x.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_y = pixel_y.unsqueeze(0).expand(batch_size, -1, -1)
        
        origin_x = origin[:, 0:1].unsqueeze(-1)  
        origin_y = origin[:, 1:2].unsqueeze(-1)
        resolution = resolution.unsqueeze(-1).unsqueeze(-1) 
        
        world_x = origin_x + pixel_x * resolution
        # Flip y-axis (image y increases downward, map y increases upward)
        world_y = origin_y + (height - 1 - pixel_y) * resolution
        
        # Stack to create [B, 2, H, W]
        coord_grid = torch.stack([world_x, world_y], dim=1)
        
        return coord_grid / self.coord_scale
    
    def forward(self, x):

        map_images = x['map_image']
        origins = x['map_origin']
        resolutions = x['map_resolution']
        
        batch_size = map_images.shape[0]
        device = map_images.device
        
        scene_grid = F.interpolate(
            map_images, 
            size=self.output_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        scene_grid = (scene_grid - 0.5) / 0.5
        
        scene_coord = self.create_coordinate_grid(
            batch_size,
            self.output_size[0], 
            self.output_size[1],
            origins,
            resolutions,
            device
        )
        
        out = x.copy() 
        out['scene/grid'] = scene_grid
        out['scene/coord'] = scene_coord
        
        return out
