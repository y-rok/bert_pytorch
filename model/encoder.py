import torch.nn as nn 
import torch.nn.functional as F
from .sub_layer.attention import MultiHeadAttention
from .sub_layer.feed_forward import FeedForward
from .embedding import InputEmb

class Encoder(nn.Module):
    def __init__(self, vocab_num, seg_num, layer_num, head_num, max_seq_len, d_model, d_k, d_ff, dropout=0.1,layernorm_eps=1e-6) -> None:
        super(Encoder,self).__init__()
        
        self.layer_num=layer_num
        self.d_model=d_model
        
        self.input_emb=InputEmb(vocab_num,seg_num,max_seq_len,d_model,dropout,layernorm_eps)
        self.attention_layers=nn.ModuleList([MultiHeadAttention(head_num,max_seq_len,d_model,d_k,dropout,layernorm_eps) for _ in range(layer_num)])
        self.feedforward_layers=nn.ModuleList([FeedForward(d_model,d_ff,layernorm_eps) for _ in range(layer_num)])
        
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
        
        inputs=self.input_emb(input_ids,seg_ids)


        att_score_list=[]
        
        for i in range(self.layer_num):
            
            # sub layer - Multi-Head Attention
            att_out, att_scores=self.attention_layers[i](inputs,masks)      
            att_score_list.append(att_scores)

            # sub layer - Point-Wise Feed Forward
            ff_out=self.feedforward_layers[i](att_out)

            if i < self.layer_num-1:
                inputs=ff_out 
            else:
                out=ff_out

        # 0으로 masking 된 input들의 최종 output embedding을 zero vector로 만듬 
        masks=masks.unsqueeze(2)
        out=out.masked_fill(masks==0,0)
        
        return out, att_score_list
            
        
            
        


