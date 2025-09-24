# Copyright 2024 The human_scene_transformer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model parameter classes.

The module contains a dataclasses used to configure the model.
"""

# from model import is_hidden_generators
from model import head, embed
# from model import scene_encoders

class ModelParams(object):
  """This object configures the model."""

  def __init__(
      self,
      human_agents_feature_config=None,
      robot_agents_feature_config=None,
      agents_position_key='human_pos',
      hidden_size=128,
      feature_embedding_size=128,
      transformer_ff_dim=64,
      ff_dim=64,
      ln_eps=1e-6,
      num_heads=4,
      num_conv_filters=(32, 32, 64, 64, 128),
      depth=8,
      mask_style=None,
      scene_encoder=None,
      prediction_head=head.LstmHead,
      prediction_head_hidden_units=None,
      pred_len=10,
      drop_prob=0.1,
      num_history_steps=12,
      num_steps=49,
      timestep=1 / 6,
  ):

    if human_agents_feature_config is None:
      self.human_agents_feature_config = {
          'human_pos': embed.AgentPositionEncoder,
          'human_kpts': embed.AgentKeypointsEncoder
          }
    
    else:
      self.human_agents_feature_config = human_agents_feature_config


    if robot_agents_feature_config is None:
      self.robot_agents_feature_config = {
          'robot_pos': embed.AgentPositionEncoder,
          }
    else:
      self.robot_agents_feature_config = robot_agents_feature_config


      
    self.agents_position_key = agents_position_key
    self.hidden_size = hidden_size
    self.feature_embedding_size = feature_embedding_size
    self.transformer_ff_dim = transformer_ff_dim
    self.ff_dim= ff_dim
    self.ln_eps = ln_eps
    self.num_heads = num_heads
    self.num_conv_filters = num_conv_filters
    self.mask_style = mask_style
    self.scene_encoder = scene_encoder
    self.prediction_head = prediction_head
    self.prediction_head_hidden_units = prediction_head_hidden_units
    self.drop_prob = drop_prob
    self.num_history_steps = num_history_steps
    self.num_steps = num_steps
    self.timestep = timestep
    self.depth = depth
    self.pred_len=pred_len