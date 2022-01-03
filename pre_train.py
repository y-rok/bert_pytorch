import argparse
from transformers import BertTokenizer
from plm_dataset import PLMDataset
from torch.utils.data import DataLoader
import json
from torch.optim import Adam
from model.encoder import Encoder
from bert import Bert
import torch.nn as nn
from utils import get_logger
import torch

logger=get_logger()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,type=str, help="model configuration file path")
    parser.add_argument('--train_path', required=True, type=str, help="training data file path")
    parser.add_argument('--steps',default=100,type=int)
    parser.add_argument("--batch_size",default=4,type=int)
    parser.add_argument("--lr",default=1e-4,help="learning rate")
    parser.add_argument("--betas",default=(0.9,0.999))
    parser.add_argument("--weight_decay",default=0.01)
    parser.add_argument("--warmup_steps",default=1000)
    parser.add_argument("--epochs",default=20)
    parser.add_argument("--gpu",default=True)

    args=parser.parse_args()

    if args.gpu and torch.cuda.is_available:
        with_cuda = True
    else:
        with_cuda = False

    with open(args.config_path,"r") as cfg_json:
        config = json.load(cfg_json)

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

    bert=Bert(config=config,tokenizer=tokenizer,with_cuda=with_cuda)
    if with_cuda: bert=bert.cuda()

    # debugging code
    # for param in bert.parameters():
    #     print(param.size())

    optim=Adam(bert.parameters(),lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)

    plm_dataset=PLMDataset(args.train_path,tokenizer,config["max_seq_len"],config["max_mask_tokens"])
    train_data_loader = DataLoader(plm_dataset, batch_size=args.batch_size)
    
    step_num=0
    loss_val=0

    sop_criterion=nn.NLLLoss()
    mlm_criterion=nn.NLLLoss(ignore_index=0)

    
    for epoch in range(args.epochs):
        
        loss_val = 0
        loss_num = 0
        for data in train_data_loader:
            if with_cuda:
                for key, item in data.items():
                    data[key]=item.cuda()

        
        sent_order_pred, masked_token_pred = bert(data["input_ids"], data["seg_ids"],data["att_masks"],data["mlm_positions"],data["mlm_masks"])
        sent_order_loss = sop_criterion(sent_order_pred,data["sop_labels"])
        masked_token_loss = mlm_criterion(masked_token_pred.transpose(1,2),data["mlm_labels"])

        loss = sent_order_loss + masked_token_loss
        # loss=sent_order_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_val+=loss.item()
        step_num+=1
        loss_num+=1

        # print("step = %d, loss = %.2f"%(step_num,loss.item()))
        
        epoch_loss = loss_val/loss_num 
        print("epoch = %d, loss = %.2f"%(epoch,epoch_loss))
        

        




        