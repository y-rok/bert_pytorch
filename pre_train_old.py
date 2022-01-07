import argparse
from transformers import BertTokenizer
from plm_dataset import PLMDataset
from torch.utils.data import DataLoader
import json
from torch.optim import Adam
from model.encoder import Encoder
from model.bert import Bert
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
from torch.optim.lr_scheduler import LambdaLR

logger=get_logger()
# logger.setLevel(logging.ERROR)

# def lambda_lr()



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
    parser.add_argument('--output_dir',required=True,type=str,help="the output path where model, vocab file will be saved")
    parser.add_argument('--steps',default=100,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--lr",default=1e-4,help="learning rate")
    parser.add_argument("--betas",default=(0.9,0.999))
    parser.add_argument("--weight_decay",default=0.01)
    parser.add_argument("--warmup_steps",default=0,type=int)
    parser.add_argument("--ft_seq_len",default=128,type=int)
    parser.add_argument("--ft_ratio",default=0.9,type=float)
    parser.add_argument("--epochs",default=1000)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("-warmup_steps",default=10000)
    parser.add_argument("--num_worklers",type=int,default=2)
    parser.add_argument("--mlm",action="store_true")
    parser.add_argument("--sop",action="store_true")
    parser.add_argument("--log_steps",type=int,default=500)
    parser.add_argument("--save_steps",type=int,default=1000)

    args=parser.parse_args()

    writer=SummaryWriter()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_model_path = os.path.join(args.output_dir,utils.MODEL_FILE_NAME)
    vocab_file_path = os.path.join(args.output_dir,utils.VOCAB_TXT_FILE_NAME)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not args.cpu and torch.cuda.is_available:
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
    vocab=bert.vocab

    if with_cuda: 
        bert=bert.cuda()
        cuda_num = torch.cuda.device_count()
        bert=nn.DataParallel(bert,list(range(cuda_num)))


    #debugging code
    for name, param in bert.named_parameters():
        if param.requires_grad:
            logger.debug("%s, %s"%(name,str(param.size())))

    

 

    
    pre_plm_dataset=PLMDataset(args.train_path,tokenizer,args.ft_seq_len,config["max_seq_len"],config["max_mask_tokens"],cached_dir=args.output_dir)
    pre_train_data_loader = DataLoader(pre_plm_dataset, batch_size=args.batch_size,num_workers=1)
    

    step_num_in_epoch = 0
    for i in pre_train_data_loader:
        step_num_in_epoch +=1
    total_step = step_num_in_epoch*args.epochs

    # print(len(pre_train_data_loader))
    post_plm_dataset=PLMDataset(args.train_path,tokenizer,config["max_seq_len"],config["max_seq_len"],config["max_mask_tokens"],cached_dir=args.output_dir)
    post_train_data_loader = DataLoader(pre_plm_dataset, batch_size=args.batch_size,num_workers=1)


    def lamda_lr(step):
        if step<args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        else:
            return max(0.0, float(total_step - step) / float(max(1, total_step - args.warmup_steps))) 



    optim=Adam(bert.parameters(),lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer=optim,lr_lambda=lamda_lr)

    sop_criterion=nn.NLLLoss()
    mlm_criterion=nn.NLLLoss(ignore_index=0)


    
    seq_len_epoch =0

    step_num=0
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

 
            result, att_score_list = bert(data)
            
            sent_order_pred = result["so_pred"]
            masked_token_pred = result["mask_pred"]

            

            sent_order_loss = sop_criterion(sent_order_pred,data["sop_labels"])
            masked_token_loss = mlm_criterion(masked_token_pred.view(-1,len(vocab)),data["mlm_labels"].view(-1))

            # loss = sent_order_loss + masked_token_loss

            #debugging code
            loss = masked_token_loss
            # loss=sent_order_loss

            
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

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


            if i%10==0:
                print("epoch = %d, step=%d loss = %.4f, sop_acc %.2f learning_rate =%.8f "%(epoch+1,step_num,mlm_loss,sop_acc,scheduler.get_last_lr()[0]))
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

  
        print("epoch = %d, step=%d, loss = %.2f, mlm_acc %.2f, %dsec (%.2fper iter) (trained max seq len = %d)"%(epoch+1,step_num,mlm_loss,mlm_acc,time.time()-start_time,(time.time()-epoch_start_time)/iter_in_epoch,seq_len_epoch))
        
  
        if epoch%10==0:
            logger.debug("seq len = %d"%(data["input_ids"][0]!=0).sum().item())
            print("saving model to %s"%output_model_path)
            torch.save(bert.state_dict(),output_model_path)
            # debugging code
            # predict_mask_token(bert,"i [MASK] i had a [MASK] answer to that question .",with_cuda=True)
        

    


        




        