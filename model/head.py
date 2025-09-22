import torch
import torch.nn as nn
import torch.nn.functional as F



class LstmHead(nn.Module):
    def __init__(self, params, num_layers=2):

        super().__init__()
        self.pred_len = params.pred_len
        self.lstm = nn.LSTM(input_size=params.hidden_size,
                           hidden_size=params.hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.mlp = nn.Linear(params.hidden_size, 2 * self.pred_len)  # Output all predictions at once
    
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        _, (hidden, _) = self.lstm(x)  # Only use final hidden state
        
        # Use the last layer's hidden state
        final_hidden = hidden[-1]  # Shape: (b*a, hidden)
        out = self.mlp(final_hidden)  # Shape: (b*a, 2*pred_len)
        # Reshape to get sequence dimension back
        preds = out.view(b, a, self.pred_len, 2)
        return preds*6.425000