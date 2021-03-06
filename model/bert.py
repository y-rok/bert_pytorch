from torch.nn.modules.linear import Linear
from transformers import BertTokenizer
import json
from model.encoder import Encoder
from torch.optim import Adam
from plm_dataset import PLMDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import get_logger

logger =get_logger()

class Bert(nn.Module):
    def __init__(self, config,tokenizer,return_sop,return_mlm,with_cuda) -> None:
        super(Bert,self).__init__()

        self.config=config
        
        self.with_cuda=with_cuda
        self.tokenizer = tokenizer
        self.vocab=self.tokenizer.get_vocab()
        self.id_to_vocab={v:k for k,v in self.vocab.items()}

        self.encoder=Encoder(len(self.vocab),self.config["seg_num"],self.config["layer_num"],self.config["head_num"],self.config["max_seq_len"],self.config["d_model"],self.config["d_k"],self.config["d_ff"],self.config["dropout"])

        self.fc_sop=nn.Linear(self.config["d_model"],2)
        self.fc_mlm=nn.Linear(self.config["d_model"],len(self.vocab))

        self.return_sop=return_sop
        self.return_mlm=return_mlm

        if self.return_mlm:
            logger.info("Enalbe Masked Laguage Modeling")
        if self.return_sop:
            logger.info("Enable Sentence Order Prediction")
        
    
    def forward(self,data):

        out, att_score_list = self.encoder(data["input_ids"], data["seg_ids"], data["att_masks"]) # [batch_size, max_seq_len, d_model]
        
        result={}
        if self.return_sop:
            result["so_pred"]=self._predict_sentence_order(out)
        if self.return_mlm:
            result["mask_pred"]=self._predict_mask_tokens(out,data["mlm_positions"],data["mlm_masks"])
            
            # debug

            # logger.debug("pred")
            # logger.debug(self.convert_mask_pred_to_token(result["mask_pred"],data["mlm_masks"],top_k=1))

        return result, att_score_list
    
    def _predict_sentence_order(self,x):
        out = self.fc_sop(x[:,0])
        return F.log_softmax(out,dim=-1)

    def _predict_mask_tokens(self,x,mlm_positions,mlm_masks):

        batch_size = mlm_positions.size()[0]

        """ batch?????? ?????? max_mask tokens ?????? MLM ????????? ?????? position?????? token output embedding??? ????????? """
        x=x.view(batch_size*self.config["max_seq_len"],self.config["d_model"]) # [batch_size*max_seq_len, d_model]

        batch_offset=torch.arange(0,batch_size*self.config["max_seq_len"],self.config["max_seq_len"]).unsqueeze(1)
        if self.with_cuda: batch_offset=batch_offset.cuda()
        
        mlm_positions+=batch_offset # [batch_size,max_mask_tokens]+[batch_size,1]
        mlm_positions=mlm_positions.view(batch_size*self.config["max_mask_tokens"])
        x=x.index_select(dim=0,index=mlm_positions)
        x=x.view(batch_size,self.config["max_mask_tokens"],self.config["d_model"]) # [batch_size,max_mask_tokens,d_model]
         
        " batch?????? ????????? position??? token output??? ???????????? Masked Token ?????? / Masking ??????"
        out = self.fc_mlm(x) 
        out = out*mlm_masks.unsqueeze(2) #  [batch_size,max_mask_tokens,vocab_num] x [batch_size, max_mask_tokens,1]
        
        out =F.log_softmax(out,dim=-1)


        
        
        return out # [batch_size, max_mask_tokens, vocan_num]

    def convert_mask_pred_to_token(self,mask_pred,mlm_masks,top_k=10):
        # a= mask_pred.topk(k=top_k, dim=2)
        mask_pred_list = mask_pred.topk(k=top_k, dim=-1).indices.tolist()
        mlm_mask_list = mlm_masks.tolist()

        result =[]
        for i, seq in enumerate(mask_pred_list):
            result_seq=[]
            for j, pred_top_k in enumerate(seq):
                if mlm_mask_list[i][j]==1:
                    result_preds=[]
                    for pred_id in pred_top_k:
                        result_preds.append(self.id_to_vocab[pred_id])
                    result_seq.append(result_preds)
            if len(result_seq)!=0:
                result.append(result_seq)
        return result

        



            
        

    