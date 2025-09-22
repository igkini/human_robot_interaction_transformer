import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentSelfAlignmentLayer(nn.Module):
    """
    Enables agent to become aware of its temporal identity.
    Agent features are cross-attended with a learned query in temporal dimension.
    """
    
    def __init__(self,
                 params,
                 name='agent_self_alignment'):
        super().__init__()
        
        self.name = name
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f'params.hidden_size ({params.hidden_size}) must be an integer '
                           f'times bigger than num_heads ({params.num_heads}).')
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=params.hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )
        
        self.attn_ln = nn.LayerNorm(params.hidden_size)
        
        self.ff_layer1 = nn.Linear(params.hidden_size, params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, params.hidden_size)

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(params.hidden_size)
        
        # Learned query vector [1, 1, 1, h]
        self.learned_query_vec = nn.Parameter(
            torch.empty(1, 1, 1, params.hidden_size).uniform_(-1.0, 1.0)
        )
    
    def build_learned_query(self, input_batch):
        """Converts self.learned_query_vec into a learned query vector."""
        # Get dimensions
        b = input_batch.shape[0]
        a = input_batch.shape[1]
        t = input_batch.shape[2]
        
        return self.learned_query_vec.repeat(b, a, t, 1)
    
    def forward(self, input_batch):
        
        b, a, t, h = input_batch.shape
        
        # Build learned query [b, a, t, h]
        learned_query = self.build_learned_query(input_batch)
        
        # Reshape tensors
        learned_query_reshaped = learned_query.view(b*a, t, h)
        input_batch_reshaped = input_batch.view(b*a, t, h)
        
        attn_out, _ = self.attn_layer(
            query=learned_query_reshaped,
            key=input_batch_reshaped,
            value=input_batch_reshaped,
            attn_mask=None,
        )
        
        attn_out = attn_out.view(b, a, t, h)
        attn_out = self.attn_ln(attn_out + input_batch)
        
        out = self.ff_layer1(attn_out)
        out = F.relu(out)
        out = self.ff_layer2(out)
        out=self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)

        return out

class SelfAttnTransformerLayer(nn.Module):
    """Performs full self-attention across the agent and time dimensions."""
    
    def __init__(
        self,
        params
    ):
        super().__init__()
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({params.hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
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
    
    def forward(self, input_batch):
        
        # [b, a, t, h]
        b, a, t, h = input_batch.shape
        
        input_batch_reshaped = input_batch.reshape(b, a * t, h)
        
        attn_out, _ = self.attn_layer(
            input_batch_reshaped,
            input_batch_reshaped,
            input_batch_reshaped,
            attn_mask=None,
        )

        attn_out = attn_out.reshape(b, a, t, h)
        attn_out = self.attn_ln(attn_out + input_batch)
        
        out = self.ff_layer1(attn_out)
        out = F.relu(out)
        out = self.ff_layer2(out)
        out = self.ff_dropout(out)
        out = self.ff_ln(out + attn_out)
        
        return out

class AgentTypeCrossAttentionLayer(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.attn_layer=nn.MultiheadAttention(
            embed_dim=params.hidden_size, 
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True)
        
        self.attn_ln = nn.LayerNorm(params.hidden_size)

        self.ff_layer1=nn.Linear(in_features=params.hidden_size, out_features=params.ff_dim)
        self.ff_layer2 = nn.Linear(params.ff_dim, params.hidden_size)

        self.ff_dropout = nn.Dropout(params.drop_prob)
        self.ff_ln = nn.LayerNorm(params.hidden_size)


    def forward(self,human_embedding, robot_embedding):
        
        b,ha,ht,hh=human_embedding.shape
        _,ra,rt,rh=robot_embedding.shape

        human_embedding=human_embedding.reshape(b,ht*ha,hh)
        robot_embedding=robot_embedding.reshape(b,rt*ra,rh)
        
        attn_out,_= self.attn_layer(
            query=human_embedding,
            key=robot_embedding,
            value=robot_embedding,
            attn_mask=None)
        
        attn_out=attn_out.reshape(b,ha,ht,hh)
        human_embedding=human_embedding.reshape(b,ha,ht,hh)
        attn_out=self.attn_ln(human_embedding+attn_out)

        out=self.ff_layer1(attn_out)
        out=F.relu(out)
        out=self.ff_layer2(out)
        out = self.ff_dropout(out)
        out=self.ff_ln(out+attn_out)

        return out