from torch import float32
import torch.nn as nn
import torch



class InputEmb(nn.Module):

    def __init__(self,vocab_num, seg_num, max_seq_len, d_model, dropout=0.1):
        super(InputEmb,self).__init__()

        self.token_emb = nn.Embedding(num_embeddings=vocab_num,embedding_dim=d_model,padding_idx=0) # google official 모델들의 vocab의 0번째 index가 [pad]임.
        self.segment_emb = nn.Embedding(num_embeddings=seg_num, embedding_dim=d_model)
        self.position_enc = PositionEnc(max_seq_len=max_seq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,input_ids,seg_ids,masks):
        """
            token embedding + segment embedding + positional embedding

        Args:
            input_ids (Tensor): [batch_size, max_seq_len]
            token_type_ids ([type]): [batch_size, max_seq_len]
        """
        
        t_emb = self.token_emb(input_ids)
        # s_emb = self.segment_emb(seg_ids).mul(masks.unsqueeze(2))
        s_emb = self.segment_emb(seg_ids)
        p_enc = self.position_enc(masks)
        return self.dropout(t_emb+s_emb+p_enc)
    
    # def _get_position_ids(self,input_size):
    #     # position id list tensor 만듬 
    #     return torch.tensor([i for i in range(input_size)],dtype=int)

class PositionEnc(nn.Module):
    
    def __init__(self, max_seq_len, d_model):
        super(PositionEnc,self).__init__()
        
        """
            Postional Embedding

                PE(pos,2i) = sin(pos/(10000^(2i/emb_dim)))
                PE(pos,2i+1) = cos(pos/(10000^(2i/emb_dim)))
        """
        self.max_seq_len=max_seq_len
        self.d_model=d_model
        pos_enc=torch.zeros((max_seq_len,d_model),dtype=torch.float32)

        pos_ids=torch.arange(0,max_seq_len,1).unsqueeze(1) # [max_seq_len,1]
        div_term = torch.pow(10000, torch.arange(0,d_model,2)/d_model) #  [emb_dim/2+1] -> 10000^(2i/emb_dim) 

        pos_enc[:,::2]= torch.sin(pos_ids/div_term) # PE(pos,2i) = sin(pos/(10000^(2i/emb_dim)))
        pos_enc[:,1::2]=torch.cos(pos_ids/div_term) # PE(pos,2i+1) = cos(pos/(10000^(2i/emb_dim)))

        self.register_buffer("pos_enc",pos_enc) # [max_seq_len,d_model]

    def forward(self,masks):
        """
        Args:
            masks (Tensor) : max_seq_len 크기로 구성된 input에서 실제 input이 존재하는 sequence 만큼 1로 그렇지 않은 pad에는 0으로 tagging
                            [batch_size, max_seq_len]

        Returns:
            [Tensor]: max_seq_len까지 0으로 padding된 positional encoding
        """
        
        # out = torch.zeros((len(seq_len_list),self.max_seq_len,self.d_model),dtype=torch.float32)
        # for index, seq_len in enumerate(seq_len_list):
        #     out[index,:,:seq_len]=self.pos_enc[:,:seq_len]
        
        # out = torch.mul(masks.unsqueeze(2),self.pos_enc) # [batch_size, max_seq_len, 1] * [max_seq_len, d_model]
        return self.pos_enc

