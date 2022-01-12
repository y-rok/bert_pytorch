import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num,max_seq_len, d_model,d_k,dropout=0.1,layernorm_eps=1e-6) -> None:
        """
            d_model - input, output의 dimension
            d_k - Key와 Query의 Dimension
            d_v - Vale의 Dimension (=d_model/head_num)

        """
        super(MultiHeadAttention,self).__init__()
        
        
        assert d_model>=head_num and d_model % head_num == 0 
        
        self.head_num=head_num
        self.d_k=d_k
        self.d_v=int(d_model/head_num)
        self.max_seq_len=max_seq_len
        
        # 모든 head의 attention을 계산하기 위한 weights
        self.query_w=nn.Linear(d_model,d_k*head_num, bias=False)
        self.key_w=nn.Linear(d_model,d_k*head_num, bias=False)
        self.value_w=nn.Linear(d_model,self.d_v*head_num, bias=False)
        
        self.dropout=nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=layernorm_eps)
        
        
        self.linear=nn.Linear(head_num*self.d_v,d_model, bias=False)
        
    def forward(self,x,masks):
        """
            concat(head_1,....,head_n)*W

        Args:
            x (Tensor): [batch_size,max_seq_len,d_model]
            masks (Tensor): [batch_size,max_seq_len]
        """
        
        batch_size = x.size()[0]
        
        """ 
            각 Head의 Scaled Dot-Product Attention 계산
                softmax(QK^T/square(d_k))*V 
        """

        # 각 Head의 query, key, value matrix 계산  
        # [batch_size, max_seq_len, d_model] x [d_model , d_k*head_num] -> [batch_size, max_seq_len, head_num * d_k] -> [batch_size, max_seq_len, head_num, d_k] -> ...
        queries = self.query_w(x).view(batch_size,self.max_seq_len,self.head_num,self.d_k).permute(0,2,1,3) # [batch_size, head_num, max_seq_len, d_k]
        keys = self.key_w(x).view(batch_size,self.max_seq_len,self.head_num,self.d_k).permute(0,2,1,3) # [batch_size, head_num, max_seq_len, d_k]
        values = self.value_w(x).view(batch_size,self.max_seq_len,self.head_num,self.d_v).permute(0,2,1,3) # [batch_size, head_num, max_seq_len, d_v]

        # 각 Head의 Attention Score 계산
        # i번쨰 token에 대한 j번째 token의 attention score
        # [max_seq_len, d_q] * [d_k, max_seq_len]
        temp = queries.matmul(keys.transpose(2,3))/(self.d_k**0.5) # [batch_size, head_num, max_seq_len, max_seq_len]
        masks = masks.view(masks.size()[0],1,1,masks.size()[1]) # [batch_size,1,1,max_seq_len]
        # masks = masks.transpose(2,3) 
        temp=temp.masked_fill(masks==0,-1e9)
        scores=F.softmax(temp,-1) # [batch_size, head_num, max_seq_len, max_seq_len]
        # scores = self.softmax(queries.matmul(keys.transpose(2,3))/self.d_k) # [batch_size, head_num, max_seq_len, max_seq_len]
        out = scores.matmul(values) # [batch_size, head_num, max_seq_len, d_v]
        
        """ 
            Multi-Head Attention 계산 
            concat(head_1,....,head_n)*W
        """
        out = out.permute(0,2,1,3).contiguous().view(batch_size,self.max_seq_len,self.head_num*self.d_v)  # [batch_size, max_seq_len, head_num*d_v]
        out=self.linear(out)
        out = self.dropout(out)
        out = out+x
        
        return self.layer_norm(out), scores # [batch_size, max_seq_len, d_model] 
        


        
        
# class SelfAttention():
#     """
#         Scaled Dot-Product Attention
#     """
#     def __init__(self, max_seq_len, d_model,d_k,d_v,dropout=0.1) -> None:
#         super(SelfAttention,self).__init__()
        
#         self.d_k=d_k
        
#         self.query_w=nn.Linear(d_model,d_k)
#         self.key_w=nn.Linear(d_model,d_k)
#         self.value_w=nn.Linear(d_model,d_v)
        
#         self.softmax=nn.Softmax(dim=1)
#         self.dropout = nn.Dropout(p=dropout)
    
#     def forward(self,x):
#         """
#             softmax(QK^T/square(d_k))*V
            
#         Args:
#             x (Tensor): [max_seq_len,d_model]
#         """
#         queries = self.query_w(x) # [max_seq_len,d_k]
#         keys = self.key_w(x) # [max_seq_len,d_k]
#         values = self.value_w(x) # [max_seq_len,d_v]
        
#         # i번쨰 token에 대한 j번째 token의 score
#         scores = self.softmax(queries.matmul(keys.transpose(0,1))/self.d_k) # [max_seq_len,max_seq_len] 
               
#         result = scores.matmul(values) # [max_seq_len,d_v]
        
        
        
#         return self.dropout(result)
        
        
        
        
        
        
               
