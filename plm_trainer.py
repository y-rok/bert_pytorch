import os
import time

import tokenizers
import utils 
import logging
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import json
from model.bert import Bert
import torch.nn as nn
from plm_dataset import PLMDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logger = utils.get_logger()

class Metric:
    def __init__(self,name,writer) -> None:
        self.reset()
        self.name = name
        self.writer=writer # tensorboard writer

    def reset(self):
        self.loss_val = 0
        self.iter_in_epoch=0
        self.correct_num= 0 
        self.loss = 0
        self.acc = 0
        self.data_num = 0

    def update_loss(self,loss_val,total_step_num):
        self.iter_in_epoch += 1
        self.loss_val+=loss_val
        self.loss=self.loss_val/self.iter_in_epoch

        self.writer.add_scalar(self.name+" Loss",self.loss,total_step_num)
    
    def update_acc(self,correct_num,data_num,step_num):
        self.correct_num+=correct_num
        self.data_num+=data_num
        self.acc=self.correct_num/self.data_num

        self.writer.add_scalar(self.name+" Acc",self.loss,step_num)

    def update_lr(self,lr):
        self.lr=lr
    
    def print_with_acc(self,step,with_acc=True):
        if with_acc:
            logger.info("step=%d loss = %.4f, acc %.2f learning_rate =%.8f "%(step,self.loss,self.acc,self.lr))
        else:
            logger.info("step=%d loss = %.4f learning_rate =%.8f "%(step,self.loss,self.lr))

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.lr = None

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
        self.lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr


class LMTrainer:
    def __init__(self, training_args, model_config) -> None:
        """
            - vocab file이 없는 경우 train_path의 corpus를 활용하여 Wordpiece로 학습 후 vocab.txt 저장
            - GPU 사용하는 경우 가용 가능한 모든 GPU 사용 (Data parallelism)
        """

        self.model_config=model_config
        self.training_args=training_args
        
        
        
        self.writer=SummaryWriter(log_dir=self.training_args.output_dir)
        
     
        self.total_step = None # 학습 수행할 총 Steps

        self.current_epoch = 0
        self.current_step = 0

        self.pre_epoch = int(self.training_args.ft_ratio*self.training_args.epochs) # tf_seq_len 길이 데이터로 학습하는 epochs
        self.post_epoch = self.training_args.epochs-self.pre_epoch # max_seq_len 길이 데이터로 학습하는 epochs
        
        self.with_cuda=None # gpu로 학습 여부

        vocab_file_path = os.path.join(training_args.output_dir,utils.VOCAB_TXT_FILE_NAME)

        assert training_args.mlm or training_args.sop, "mlm or sop training_args must be identified"

        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        
        if not training_args.cpu and torch.cuda.is_available:
            self.with_cuda = True
        else:
            self.with_cuda = False
        
        """ vocab file이 없는 경우 train_path의 corpus를 활용하여 Wordpiece로 학습 후 vocab.txt 저장 """
        if not os.path.exists(vocab_file_path):
            logger.info("Training wordpiece tokenizer because %s does not exist"%vocab_file_path)
            self._train_tokenizer(training_args.train_path, training_args.output_dir)
        else:
            logger.info("Loading vocab file in %s")

        tokenizer=BertTokenizer(vocab_file=vocab_file_path,do_lower_case=True)
        self.vocab=tokenizer.vocab

        """" BERT 모델 초기화 """
        self.bert=Bert(config=model_config,tokenizer=tokenizer,with_cuda=self.with_cuda)

        if self.with_cuda: 
            self.bert=self.bert.cuda()
            cuda_num = torch.cuda.device_count()
            self.bert=nn.DataParallel(self.bert,list(range(cuda_num)))

        # debug - 모델 정보 출력
        if training_args.debug:
              for name, param in self.bert.named_parameters():
                if param.requires_grad:
                    logger.debug("%s, %s"%(name,str(param.size())))

        """ Dataset, DataLoader 초기화"""

        # Bert base의 경우 128 token으로 90% 학습 후 512 token으로 마지막 10% 학습
        # max_seq_len 보다 짧은 seq_len으로 학습하기 위한 데이터셋
        pre_plm_dataset=PLMDataset(training_args.train_path,tokenizer,training_args.ft_seq_len,model_config["max_seq_len"],model_config["max_mask_tokens"],cached_dir=training_args.output_dir)
        self.pre_train_data_loader = DataLoader(pre_plm_dataset, batch_size=training_args.batch_size,num_workers=training_args.num_workers)
    
        # 이후 max_seq_len으로 학습하기 위한 Dataset
        post_plm_dataset=PLMDataset(training_args.train_path,tokenizer,model_config["max_seq_len"],model_config["max_seq_len"],model_config["max_mask_tokens"],cached_dir=training_args.output_dir)
        self.post_train_data_loader = DataLoader(post_plm_dataset, batch_size=training_args.batch_size,num_workers=training_args.num_workers)

        # 학습 끝까지의 step 수 self.total_step를 계산
        # self._calcuate_total_step(self.pre_train_data_loader,self.post_train_data_loader)

        """ Optimzer, criterion 초기화 """
        self.optim=Adam(self.bert.parameters(),lr=self.training_args.lr,betas=self.training_args.betas, weight_decay=self.training_args.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, model_config["d_model"], n_warmup_steps=self.training_args.warmup_steps)
        # self.optim=Adam(self.bert.parameters(),lr=training_args.lr,betas=training_args.betas, weight_decay=training_args.weight_decay)
        # self.scheduler = LambdaLR(optimizer=self.optim,lr_lambda=self._lamda_lr)

        if training_args.mlm:
            self.mlm_criterion=nn.NLLLoss(ignore_index=0)
            self.mlm_metric=Metric("MLM/Train",self.writer)
        elif training_args.sop:
            self.sop_criterion=nn.NLLLoss()
            self.sop_metric=Metric("SOP/Train",self.writer)


        
    def train(self):

        logger.info("Start Training!")
        for epoch in range(self.training_args.epochs):

            self.current_epoch+=1

            # ft ratio 만큼 짧은 seq length로 학습 후 나머지 부분은 max_seq_len로 학습 (수렴 속도를 높이기 위해)
            if self.current_epoch<self.pre_epoch:
                train_data_loader=self.pre_train_data_loader
                seq_len_epoch=self.training_args.ft_seq_len
            else:
                train_data_loader=self.post_train_data_loader
                seq_len_epoch=self.model_config["max_seq_len"]
            
            start_time=time.time()
            iter_num=0
            for i, data in enumerate(train_data_loader):
                self._step(data)

                # log_steps 마다 출력
                if self.current_step!=0 and self.current_step % self.training_args.log_steps==0:
                    self._print_metric()
                if self.current_step!=0 and self.current_step % self.training_args.save_steps==0:
                    self._save_model(self.current_step)

                iter_num+=i
            
            # epoch 단위 출력
            elapsed_time = time.time()-start_time
            logger.info("========================================")
            logger.info("%d epoch summary - data max seq len = %d elapsed time = %dsec (%.2fsec per iter)"%(self.current_epoch,seq_len_epoch,elapsed_time,elapsed_time/iter_num))
            self._print_metric()
            logger.info("========================================")
            
            if self.training_args.mlm:
                self.mlm_metric.reset()
            if self.training_args.sop:
                self.sop_metric.reset()
        
    def _save_model(self,steps):

        output_model_path= os.path.join(self.training_args.output_dir,"checkpoint_"+str(steps)+".pt")

        print("saving model to %s"%output_model_path)
        torch.save(self.bert.module.state_dict(),output_model_path)

    def _step(self,data):

        self.current_step+=1
        
        if self.with_cuda:
            for key, item in data.items():
                data[key]=item.cuda()
        
        # 추론
        result, att_score_list = self.bert(data,return_sop=self.training_args.sop,return_mlm=self.training_args.mlm)

        # loss 계산
        loss=None
        if self.training_args.sop:
            sop_loss=self.sop_criterion(result["so_pred"],data["sop_labels"])
            loss = sop_loss 
        elif self.training_args.mlm:
            mlm_loss=self.mlm_criterion(result["mask_pred"].transpose(1,2),data["mlm_labels"])
            loss=mlm_loss if loss == None else loss+mlm_loss

        # gradient 계산 및 update
        # self.optim.zero_grad()
        # loss.backward()
        # self.optim.step()
        self.optim_schedule.zero_grad()
        loss.backward()
        self.optim_schedule.step_and_update_lr()
        
        batch_size = data["input_ids"].size()[0]
        # metric 계산
        if self.training_args.mlm:
            mlm_correct_num = result["mask_pred"].argmax(dim=-1).eq(data["mlm_labels"]).sum().item()-(data["mlm_masks"]==0).sum().item()
            mlm_mask_num = (data["mlm_masks"]==1).sum().item()
            self.mlm_metric.update_loss(mlm_loss.item(),self.current_step)
            self.mlm_metric.update_acc(mlm_correct_num,mlm_mask_num,self.current_step)
            self.mlm_metric.update_lr(self.optim_schedule.lr)
        if self.training_args.sop:
            sop_correct_num = result["so_pred"].argmax(dim=-1).eq(data["sop_labels"]).sum().item()
            sop_num = data["sop_labels"].size()[0]
            self.sop_metric.update_loss(sop_loss.item(),self.current_step)
            self.sop_metric.update_acc(sop_correct_num,sop_correct_num,self.current_step)
            self.sop_metric.update_lr(self.optim_schedule.lr)


        # self.scheduler.step() #learning rate update

    
    def _print_metric(self):
        if self.training_args.mlm:
            self.mlm_metric.print_with_acc(self.current_step)

        if self.training_args.sop:
            self.sop_metric.print_with_acc(self.current_step)

    def _calcuate_total_step(self,pre_train_data_loader,post_train_data_loader):

        # iteration을 끝까지 수행하여 epoch 당 step 수 측정
        step_num_in_pre_epoch = 0
        for _ in pre_train_data_loader:
            step_num_in_pre_epoch +=1
        self.total_step = step_num_in_pre_epoch*self.pre_epoch

        step_num_in_post_epoch = 0
        for _ in post_train_data_loader:
            step_num_in_post_epoch+=1
        self.total_step = step_num_in_pre_epoch*self.post_epoch


    def _lamda_lr(self,step):
        if step<self.training_args.warmup_steps:
            return float(step) / float(max(1, self.training_args.warmup_steps))
        else:
            return max(0.0, float(self.total_step - step) / float(max(1, self.total_step - self.training_args.warmup_steps))) 



    def _train_tokenizer(self,train_path,output_dir,vocab_size=32000, min_freq=3):
        """
            train_path의 학습데이터를 활용하여 Wordpice 학습 및 Vocab 생성
                - Special Token ([PAD], [MASK], [CLS]....)들 추가
        """
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
            show_progress=True,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )

        vocab_path = os.path.join(output_dir,utils.VOCAB_FILE_NAME)
        vocab_file = os.path.join(output_dir,utils.VOCAB_TXT_FILE_NAME)
        
        tokenizer.save(vocab_path,True)

        logger.info("Writing vocab file to %s"%vocab_file)
        f = open(vocab_file,'w',encoding='utf-8')
        with open(vocab_path) as json_file:
            json_data = json.load(json_file)
            for item in json_data["model"]["vocab"].keys():
                f.write(item+'\n')

            f.close()

    

    