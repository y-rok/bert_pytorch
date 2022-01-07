import torch.nn as nn 
import torch.nn.functional as F
from .sub_layer.attention import MultiHeadAttention
from .sub_layer.feed_forward import FeedForward
from .embedding import InputEmb

class Encoder(nn.Module):
    def __init__(self, vocab_num, seg_num, layer_num, head_num, max_seq_len, d_model, d_k, d_ff, dropout=0.1) -> None:
        super(Encoder,self).__init__()
        
        self.layer_num=layer_num
        self.d_model=d_model
        
        self.input_emb=InputEmb(vocab_num,seg_num,max_seq_len,d_model,dropout)
        self.attention_layers=nn.ModuleList([MultiHeadAttention(head_num,max_seq_len,d_model,d_k,dropout) for _ in range(layer_num)])
        self.feedforward_layers=nn.ModuleList([FeedForward(d_model,d_ff,dropout) for _ in range(layer_num)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self,input_ids,seg_ids, masks):
        """
            Layer = Input Embedding -> Multi head attention -> LayerNorm(x + Sublayer(x)) -> point-wise feed forward -> aLayerNorm(x + Sublayer(x))
            
        Args:
            int_ids (Tensor): input id들의 sequence로 구성, max_seq_len보다 짧은 sequence의 경우 나머지 0으로 padding
                                [batch_size, max_seq_len]
            seg_ids (Tensor): segment id들의 sequence로 구성, max_seq_len보다 짧은 sequence의 경우 나머지 0으로 padding
                                [batch_size, max_seq_len]
            masks (Tensor): sequence 내에서 input이 있는 경우 1, 그렇지 않은 경우 0으로 masking
                                [batch_size, max_seq_len]
        """
        
        inputs=self.input_emb(input_ids,seg_ids,masks)
        inputs=self.layer_norm(inputs)

        att_score_list=[]
        
        for i in range(self.layer_num):
            
            # sub layer - Multi-Head Attention
            temp, att_scores=self.attention_layers[i](inputs,masks)
            att_out=self.layer_norm(inputs+temp) # LayerNorm(x + Sublayer(x))
            
            att_score_list.append(att_scores)

            # sub layer - Point-Wise Feed Forward
            temp=self.feedforward_layers[i](att_out)
            ff_out=self.layer_norm(att_out+temp) # LayerNorm(x + Sublayer(x))
            
            if i < self.layer_num-1:
                inputs=ff_out 
            else:
                out=ff_out

        # 0으로 masking 된 input들의 최종 output embedding을 zero vector로 만듬 
        masks=masks.unsqueeze(2)
        out=out.masked_fill(masks==0,0)
        
        return out, att_score_list
            
        
            
        


