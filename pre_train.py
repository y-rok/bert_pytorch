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
from itertools import chain
from tokenizers import BertWordPieceTokenizer
import os
import utils 
from torch.utils.tensorboard import SummaryWriter
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

def train_tokenizer(train_path,output_dir,vocab_size=32000, min_freq=3):

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True, 
        lowercase=True,
        wordpieces_prefix="##"
    )

    tokenizer.train(
        files=[train_path],
        limit_alphabet=6000,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        # pair가 5회이상 등장할시에만 학습
        show_progress=True,
        # 진행과정 출력 여부
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    vocab_path = os.path.join(output_dir,utils.VOCAB_FILE_NAME)
    vocab_file = os.path.join(output_dir,utils.VOCAB_TXT_FILE_NAME)

    
    tokenizer.save(vocab_path,True)


    f = open(vocab_file,'w',encoding='utf-8')
    with open(vocab_path) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')

        f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,type=str, help="model configuration file path")
    parser.add_argument('--train_path', required=True, type=str, help="training data file path")
    parser.add_argument('--output_dir',required=True,type=str,help="model output path")
    parser.add_argument('--steps',default=100,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--lr",default=1e-4,help="learning rate")
    parser.add_argument("--betas",default=(0.9,0.999))
    parser.add_argument("--weight_decay",default=0.01)
    parser.add_argument("--warmup_steps",default=1000)
    parser.add_argument("--ft_seq_len",default=128,type=int)
    parser.add_argument("--ft_ratio",default=0.9,type=float)
    parser.add_argument("--epochs",default=1000)
    parser.add_argument("--gpu",default=True)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("-warmup_steps",default=10000)

    args=parser.parse_args()

    writer=SummaryWriter()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_model_path = os.path.join(args.output_dir,utils.MODEL_FILE_NAME)
    vocab_file_path = os.path.join(args.output_dir,utils.VOCAB_TXT_FILE_NAME)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.gpu and torch.cuda.is_available:
        with_cuda = True
    else:
        with_cuda = False

    with open(args.config_path,"r") as cfg_json:
        config = json.load(cfg_json)
    
    if not os.path.exists(vocab_file_path):
        print("training wordpiece tokenizer")
        train_tokenizer(args.train_path, args.output_dir)

    # tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer=BertTokenizer(vocab_file=vocab_file_path,do_lower_case=True)

    bert=Bert(config=config,tokenizer=tokenizer,with_cuda=with_cuda)
    if with_cuda: 
        bert=bert.cuda()
        cuda_num = torch.cuda.device_count()
        bert=nn.DataParallel(bert,list(range(cuda_num)))


    #debugging code
    for name, param in bert.named_parameters():
        if param.requires_grad:
            logger.debug("%s, %s"%(name,str(param.size())))

    optim=Adam(bert.parameters(),lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)
    optim_schedule = ScheduledOptim(optim, config["d_model"], n_warmup_steps=args.warmup_steps)

    pre_plm_dataset=PLMDataset(args.train_path,tokenizer,args.ft_seq_len,config["max_seq_len"],config["max_mask_tokens"],cached_dir=args.output_dir)
    pre_train_data_loader = DataLoader(pre_plm_dataset, batch_size=args.batch_size,num_workers=1)
    
    post_plm_dataset=PLMDataset(args.train_path,tokenizer,config["max_seq_len"],config["max_seq_len"],config["max_mask_tokens"],cached_dir=args.output_dir)
    post_train_data_loader = DataLoader(pre_plm_dataset, batch_size=args.batch_size,num_workers=1)

    sop_criterion=nn.NLLLoss()
    mlm_criterion=nn.NLLLoss(ignore_index=0)


    step_num=0
    seq_len_epoch =0

    start_time=time.time()
    
    for epoch in range(args.epochs):
        
        loss_val = 0
        sop_correct = 0
        data_num=0
        iter_in_epoch=0

        mlm_correct=0
        mlm_num=0

        sop_loss=0
        mlm_loss=0

        epoch_start_time=time.time()
        # ft ratio 만큼 짧은 seq length로 학습 후 나머지 부분은 max_seq_len로 학습 (수렴 속도를 높이기 위해)
        if epoch<args.ft_ratio*args.epochs:
            train_data_loader=pre_train_data_loader
            seq_len_epoch=args.ft_seq_len
        else:
            train_data_loader=post_train_data_loader
            seq_len_epoch=config["max_seq_len"]

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

            mlm_correct+=masked_token_pred.argmax(dim=-1).eq(data["mlm_labels"]).sum().item()-(data["mlm_masks"]==0).sum().item()
            sop_correct+=sent_order_pred.argmax(dim=-1).eq(data["sop_labels"]).sum().item()

            mlm_num+=(data["mlm_masks"]==1).sum().item()
            data_num+=data["input_ids"].size()[0]
            iter_in_epoch+=1
            step_num+=1
            
            mlm_loss = loss_val/iter_in_epoch 
            mlm_acc = mlm_correct/mlm_num
            sop_acc = sop_correct/data_num

            
            

            if i%1000==0:
                print("epoch = %d, loss = %.2f, sop_acc %.2f (iteration = %d)"%(epoch,mlm_loss,sop_acc,i))
                #debugging code
                # logger.debug("label")
                # label_list=[]
                # mlm_masks_list=data["mlm_masks"].tolist()
                # mlm_label_list=data["mlm_labels"].tolist()
                # for index_1,item in enumerate(mlm_label_list):
                #     for index_2,value in enumerate(item):
                #         if mlm_masks_list[index_1][index_2]==1:
                #             label_list.append(bert.id_to_vocab[value])
                # logger.debug(label_list)

                # logger.debug("pred")
                # pred_vocabs=bert.convert_mask_pred_to_token(masked_token_pred,data["mlm_masks"],top_k=1)
                # pred_vocabs=list(chain(*list(chain(*pred_vocabs))))
                # logger.debug(pred_vocabs)


        writer.add_scalar("MLM Loss/train",mlm_loss,step_num)
        writer.add_scalar("MLM Acc/train",mlm_acc,step_num)
        # writer.add_scalar("sop_acc/train",sop_acc,epoch)

  
        print("epoch = %d, step=%d loss = %.2f, mlm_acc %.2f, %dsec(%.2fper iter) (trained max seq len = %d)"%(epoch+1,step_num,mlm_loss,mlm_acc,time.time()-start_time,(time.time()-epoch_start_time)/iter_in_epoch,seq_len_epoch))
        
  
        if epoch%10==0:
            logger.debug("seq len = %d"%(data["input_ids"][0]!=0).sum().item())
            print("saving model to %s"%output_model_path)
            torch.save(bert.state_dict(),output_model_path)
            # debugging code
            # predict_mask_token(bert,"i [MASK] i had a [MASK] answer to that question .",with_cuda=True)
        

    


        




        