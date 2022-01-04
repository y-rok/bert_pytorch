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
import logging
from test import predict_mask_token
import numpy as np
import time

logger=get_logger()
# logger.setLevel(logging.ERROR)


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,type=str, help="model configuration file path")
    parser.add_argument('--train_path', required=True, type=str, help="training data file path")
    parser.add_argument('--output_path',required=True,type=str,help="model output path")
    parser.add_argument('--steps',default=100,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--lr",default=1e-4,help="learning rate")
    parser.add_argument("--betas",default=(0.9,0.999))
    parser.add_argument("--weight_decay",default=0.01)
    parser.add_argument("--warmup_steps",default=1000)
    parser.add_argument("--epochs",default=20)
    parser.add_argument("--gpu",default=True)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("-warmup_steps",default=10000)

    args=parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.gpu and torch.cuda.is_available:
        with_cuda = True
    else:
        with_cuda = False

    with open(args.config_path,"r") as cfg_json:
        config = json.load(cfg_json)

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

    bert=Bert(config=config,tokenizer=tokenizer,with_cuda=with_cuda)
    if with_cuda: bert=bert.cuda()


    #debugging code
    for name, param in bert.named_parameters():
        if param.requires_grad:
            logger.debug("%s, %s"%(name,str(param.size())))

    optim=Adam(bert.parameters(),lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)
    optim_schedule = ScheduledOptim(optim, config["d_model"], n_warmup_steps=args.warmup_steps)

    plm_dataset=PLMDataset(args.train_path,tokenizer,config["max_seq_len"],config["max_mask_tokens"])
    train_data_loader = DataLoader(plm_dataset, batch_size=args.batch_size,num_workers=1)
    
    step_num=0
    loss_val=0

    sop_criterion=nn.NLLLoss()
    mlm_criterion=nn.NLLLoss(ignore_index=0)

    min_loss =None

    
    for epoch in range(args.epochs):
        
        loss_val = 0
        loss_num = 0

        sop_correct = 0
        sop_total_num=0

        mlm_correct=0
        mlm_total_num=0

        for i, data in enumerate(train_data_loader):
            if with_cuda:
                for key, item in data.items():
                    data[key]=item.cuda()

        
            result = bert(data)
            
            sent_order_pred = result["so_pred"]
            masked_token_pred = result["mask_pred"]

            sent_order_loss = sop_criterion(sent_order_pred,data["sop_labels"])
            masked_token_loss = mlm_criterion(masked_token_pred.transpose(1,2),data["mlm_labels"])

            loss = sent_order_loss + masked_token_loss

            #debugging code
            loss = masked_token_loss
            # loss=sent_order_loss

            optim_schedule.zero_grad()
            loss.backward()
            optim_schedule.step_and_update_lr()

            loss_val+=loss.item()
            step_num+=1
            loss_num+=1

            sop_correct+=sent_order_pred.argmax(dim=-1).eq(data["sop_labels"]).sum().item()
            sop_total_num+=sent_order_pred.size()[0]


            # print("step = %d, loss = %.2f"%(step_num,loss.item()))
            
            epoch_loss = loss_val/loss_num 
            sop_acc = sop_correct/sop_total_num

            if i%100==0:
                print("epoch = %d, loss = %.2f, sop_acc %.2f (iteration = %d)"%(epoch,epoch_loss,sop_acc,i),end="\r")
        


        print("epoch = %d, loss = %.2f, sop_acc %.2f"%(epoch,epoch_loss,sop_acc))
        
        if min_loss==None or min_loss>epoch_loss:
            min_loss=epoch_loss
            print("saving model to %s"%args.output_path)
            torch.save(bert.state_dict(),args.output_path)
            # debugging code
            predict_mask_token(bert,"i [MASK] i had a [MASK] answer to that question .",with_cuda=True)
        

    


        




        