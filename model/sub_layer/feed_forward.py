import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    
    def __init__(self,d_model,d_ff,dropout=0.1) -> None:
        super(FeedForward,self).__init__()
        
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x):
        """
            fc2(relu(fc1(x)))
        Args:
            x (Tensor): [batch_size, max_seq_len, d_model] 
        """
        # out = F.relu(self.fc1(x))
        out = F.gelu(self.fc1(x))
        out = self.fc2(out)
        
        return self.dropout(out)
        