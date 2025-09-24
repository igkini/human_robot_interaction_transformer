import torch.nn as nn
from model.model_params import ModelParams
import torch.nn as nn

from model import agent_encoder, attention, scene_encoder,scene_attn, head, preprocess

class HumanRobotInteractionTransformer(nn.Module):

    def __init__(
        self,
        params: ModelParams,
    ):
        super().__init__()

        self.params = params

        self.human_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer('human',params)
        self.robot_agent_encoding_layer = agent_encoder.FeatureConcatAgentEncoderLayer('robot',params)
        self.human_self_alignment_layer = attention. AgentSelfAlignmentLayer(params, seq_downsample=1)
        self.agent_type_cross_attn_layer= attention.AgentTypeCrossAttentionLayer(params)
        self.human_self_attn_layer= attention.SelfAttnTransformerLayer(params)

        self.scene_preprocess_layer=preprocess.GridPreprocessLayer()
        self.scene_encoder_layer= scene_encoder.ConvOccupancyGridEncoderLayer(params)
        self.scene_cross_attn_layer=scene_attn.SceneCrossAttnTransformerLayer(params)
        self.prediction_head_layer=head.LstmHead(params)


    # --------------------------------------------------------------------- #
    def forward(self, input_batch):
        
        human_emb = self.human_agent_encoding_layer(input_batch)
        robot_emb = self.robot_agent_encoding_layer(input_batch)

        out= self.human_self_alignment_layer(human_emb)
        out=self.agent_type_cross_attn_layer(out,robot_emb)
        out=self.human_self_attn_layer(out)

        scene_enc=self.scene_preprocess_layer(input_batch)
        scene_enc = self.scene_encoder_layer(scene_enc)
        out = self.scene_cross_attn_layer(out,scene_enc)
        out = self.prediction_head_layer(out)

        return out