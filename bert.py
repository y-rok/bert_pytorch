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

class Bert(nn.Module):
    def __init__(self, config,tokenizer,with_cuda) -> None:
        super(Bert,self).__init__()

        self.config=config
        
        self.with_cuda=with_cuda
        self.tokenizer = tokenizer
        self.vocab=self.tokenizer.get_vocab()
        self.id_to_vocab={v:k for k,v in self.vocab.items()}


        self.encoder=Encoder(len(self.vocab),self.config["seg_num"],self.config["layer_num"],self.config["head_num"],self.config["max_seq_len"],self.config["d_model"],self.config["d_k"],self.config["d_ff"],self.config["dropout"])

        self.fc_sop=nn.Linear(self.config["d_model"],2)
        self.fc_mlm=nn.Linear(self.config["d_model"],len(self.vocab))
    
    def forward(self,data,return_sop=True,return_mlm=True):

        out = self.encoder(data["input_ids"], data["seg_ids"], data["att_masks"]) # [batch_size, max_seq_len, d_model]
        
        result={}
        if return_sop:
            result["so_pred"]=self._predict_sentence_order(out)
        if return_mlm:
            result["mask_pred"]=self._predict_mask_tokens(out,data["mlm_positions"],data["mlm_masks"])
        return result
    
    def _predict_sentence_order(self,x):
        out = self.fc_sop(x[:,0])
        return F.log_softmax(out,dim=1)

    def _predict_mask_tokens(self,x,mlm_positions,mlm_masks):

        batch_size = mlm_positions.size()[0]

        """ batch별로 최대 max_mask tokens 개씩 MLM 예측을 위한 position들의 token output embedding을 가져옴 """
        x=x.view(batch_size*self.config["max_seq_len"],self.config["d_model"]) # [batch_size*max_seq_len, d_model]

        batch_offset=torch.arange(0,batch_size*self.config["max_seq_len"],self.config["max_seq_len"]).unsqueeze(1)
        if self.with_cuda: batch_offset=batch_offset.cuda()
        
        mlm_positions+=batch_offset # [batch_size,max_mask_tokens]+[batch_size,1]
        mlm_positions=mlm_positions.view(batch_size*self.config["max_mask_tokens"])
        x=x.index_select(0,mlm_positions)
        x=x.view(batch_size,self.config["max_mask_tokens"],self.config["d_model"]) # [batch_size,max_mask_tokens,d_model]
         
        " batch별로 예측한 position의 token output을 활용하여 Masked Token 예측 / Masking 적용"
        out = self.fc_mlm(x)
        out = out*mlm_masks.unsqueeze(2)
        out =F.log_softmax(out,dim=2)
        
        return out # [batch_size, max_mask_tokenxs, vocan_num]

    def convert_mask_pred_to_token(self,mask_pred,mlm_masks,top_k=3):
        # a= mask_pred.topk(k=top_k, dim=2)
        mask_pred_list = mask_pred.topk(k=top_k, dim=2).indices.tolist()
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

        
    # def __init__(self,config_path) -> None:

    #     with open(config_path,"r") as cfg_json:
    #         self.config = json.load(cfg_json)
        
    #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     self.encoder=Encoder(len(self.tokenizer.get_vocab()),self.config["seg_num"],self.config["layer_num"],self.config["head_num"],self.config["max_seq_len"],self.config["d_model"],self.config["d_k"],self.config["d_ff"],self.config["dropout"])

    # def predict_tokens(self):

    # def get_mlm_loss(self,out,labels):

    #     return loss
    # def pre_train(self,train_path,args):
        
    #     optim=Adam(self.encoder.get_parameter(),lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)

    #     plm_dataset=PLMDataset(args.train_path,self.tokenizer,self.config["max_seq_len"])
    #     train_data_loader = DataLoader(plm_dataset, batch_size=args.batch_size)

    #     loss = nn.NLLLoss()

    #     for input_ids, seg_ids, att_masks, labels, correct_order in train_data_loader:
    #         out = self.encoder(input_ids, seg_ids, att_masks)




            
        

    