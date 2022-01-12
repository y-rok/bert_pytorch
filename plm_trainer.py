import os
import time

import copy
import utils 
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import json
from model.bert import Bert
import torch.nn as nn
from plm_dataset import PLMDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import Metric
from transformers import get_linear_schedule_with_warmup

logger = utils.get_logger()


class LMTrainer:
    def __init__(self, training_args, model_config) -> None:
        """
            - vocab file이 없는 경우 train_path의 corpus를 활용하여 Wordpiece로 학습 후 vocab.txt 저장
            - GPU 사용하는 경우 가용 가능한 모든 GPU 사용 (Data parallelism)
                - 특정 GPU만 사용시 shell 에서 export CUDA_VISIBLE_DEVICES 사용
        """

        self.model_config=model_config
        self.training_args=training_args

        
        self.writer=SummaryWriter(log_dir=self.training_args.output_dir)
        
     
        self.total_steps = None # 학습 수행할 총 Steps

        self.current_epoch = 0
        self.current_step = 0

        self.pre_epoch = int(self.training_args.ft_ratio*self.training_args.epochs) # tf_seq_len 길이 데이터로 학습하는 epochs
        self.post_epoch = self.training_args.epochs-self.pre_epoch # max_seq_len 길이 데이터로 학습하는 epochs
        
        self.with_cuda=None # gpu로 학습 여부

        vocab_file_path = os.path.join(training_args.output_dir,utils.VOCAB_TXT_FILE_NAME)

        assert training_args.mlm or training_args.sop, "mlm or sop training_args must be identified"

        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)

        training_args_path = os.path.join(training_args.output_dir,utils.TRAINING_ARGS_NAME)
        logger.info("Writing training args file to %s"%training_args_path)
        with open(training_args_path,"w") as f:
            json.dump(vars(training_args),f)
        
        log_file_path = os.path.join(training_args.output_dir,utils.TRAINING_LOG_NAME)
        logger.info("Writing Log file to %s"%log_file_path)
        utils.add_handler_to_logger(logger, log_file_path)

        model_config_path = os.path.join(training_args.output_dir,utils.MODEL_CONFIG_NAME)
        logger.info("Writing model configuration file to %s"%model_config_path)
        with open(model_config_path,"w") as f:
            json.dump(model_config,f)

        
        if not training_args.cpu and torch.cuda.is_available:
            self.with_cuda = True
        else:
            self.with_cuda = False

        
        
        """ vocab file이 없는 경우 train_path의 corpus를 활용하여 Wordpiece로 학습 후 vocab.txt 저장 """
        if not os.path.exists(vocab_file_path):
            logger.info("Training wordpiece tokenizer because %s does not exist"%vocab_file_path)
            self._train_tokenizer(training_args.train_path, training_args.output_dir)
        else:
            logger.info("Loading vocab file in %s"%vocab_file_path)

        tokenizer=BertTokenizer(vocab_file=vocab_file_path,do_lower_case=True)
        self.vocab=tokenizer.vocab
        self.id_to_vocab = {v:k for k,v in self.vocab.items()}

        """" BERT 모델 초기화 """
        self.bert=Bert(config=model_config,tokenizer=tokenizer,return_sop=self.training_args.sop, return_mlm=self.training_args.mlm,with_cuda=self.with_cuda)

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
        pre_plm_dataset=PLMDataset(training_args.train_path,tokenizer,training_args.ft_seq_len,model_config["max_seq_len"],model_config["max_mask_tokens"],cached_dir=training_args.output_dir,mlm_data=self.training_args.mlm,sop_data=self.training_args.sop)
        self.pre_train_data_loader = DataLoader(pre_plm_dataset, batch_size=training_args.batch_size,num_workers=training_args.num_workers)
    
        # 이후 max_seq_len으로 학습하기 위한 Dataset
        post_plm_dataset=copy.deepcopy(pre_plm_dataset)
        post_plm_dataset._set_data_max_seq_len(model_config["max_seq_len"])
        # post_plm_dataset=PLMDataset(training_args.train_path,tokenizer,model_config["max_seq_len"],model_config["max_seq_len"],model_config["max_mask_tokens"],cached_dir=training_args.output_dir,mlm_data=self.training_args.mlm,sop_data=self.training_args.sop)
        self.post_train_data_loader = DataLoader(post_plm_dataset, batch_size=training_args.batch_size,num_workers=training_args.num_workers)

        self._calculate_steps()


        """ Optimzer, criterion 초기화 """
        self.optim=AdamW(self.bert.parameters(),lr=self.training_args.lr,betas=self.training_args.betas, weight_decay=self.training_args.weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, model_config["d_model"], n_warmup_steps=self.training_args.warmup_steps)
        self.scheduler=get_linear_schedule_with_warmup(self.optim,self.warmup_steps,self.total_steps) # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

        if training_args.mlm:
            self.mlm_criterion=nn.NLLLoss(ignore_index=0)
            self.mlm_metric=Metric("MLM/Train",self.writer)
        if training_args.sop:
            self.sop_criterion=nn.NLLLoss()
            self.sop_metric=Metric("SOP/Train",self.writer)

        logger.info("==============================================")
        logger.info("<Training Arguments>")
        logger.info(self.training_args)
        logger.info("==============================================")
        logger.info("<Model Config>")
        logger.info(model_config)

        
    def train(self):

        logger.info("epoch = %d, total step = %d, warm up steps = %d"%(self.training_args.epochs,self.total_steps,self.warmup_steps))
        logger.info("Start Training!")
        
        start_time=time.time()

        for epoch in range(self.training_args.epochs):

            self.current_epoch+=1

            # ft ratio 만큼 짧은 seq length로 학습 후 나머지 부분은 max_seq_len로 학습 (수렴 속도를 높이기 위해)
            if self.current_epoch<self.pre_epoch:
                train_data_loader=self.pre_train_data_loader
                seq_len_epoch=self.training_args.ft_seq_len
            else:
                train_data_loader=self.post_train_data_loader
                seq_len_epoch=self.model_config["max_seq_len"]
            
            
    
            for i, data in enumerate(train_data_loader):
                self._step(data)

                # log_steps 마다 출력
                if self.current_step!=0 and self.current_step % self.training_args.log_steps==0:
                    self._print_metric()
                if self.current_step!=0 and self.current_step % self.training_args.save_steps==0:
                    self._save_model(self.current_step)

            
            # epoch 단위 출력
            elapsed_time = time.time()-start_time
            logger.info("========================================")
            logger.info("%d epoch summary - data max seq len = %d elapsed time = %dsec"%(self.current_epoch,seq_len_epoch,elapsed_time))
            self._print_metric()
            logger.info("========================================")
            
            if self.training_args.mlm:
                self.mlm_metric.reset()
            if self.training_args.sop:
                self.sop_metric.reset()
                
        self._save_model(self.current_step) # 마지막 모델 저장
        
    def _save_model(self,steps):

        output_model_path= os.path.join(self.training_args.output_dir,"checkpoint_"+str(steps)+".pt")

        logger.info("saving model to %s"%output_model_path)
        torch.save(self.bert.module.state_dict(),output_model_path)

    def _step(self,data):

        self.current_step+=1
        
        if self.with_cuda:
            for key, item in data.items():
                data[key]=item.cuda()
        
        # 추론
        result, att_score_list = self.bert(data)
        batch_size = data["input_ids"].size()[0]
        # loss 계산
        loss=None
        if self.training_args.sop:
            sop_loss=self.sop_criterion(result["so_pred"],data["sop_labels"])
            loss = sop_loss 
        if self.training_args.mlm:
            # mlm_loss=self.mlm_criterion(result["mask_pred"].transpose(1,2),data["mlm_labels"]) # NLL loss input : (N,C,d_1), target : (N,d_1)
            mlm_loss=self.mlm_criterion(result["mask_pred"].view(batch_size*self.model_config["max_mask_tokens"],-1),data["mlm_labels"].view(-1))
            loss=mlm_loss if loss == None else loss+mlm_loss

        # gradient 계산 및 update
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        
        # metric 계산
        if self.training_args.mlm:
            mlm_correct_num = result["mask_pred"].argmax(dim=-1).eq(data["mlm_labels"]).sum().item()-(data["mlm_masks"]==0).sum().item()
            mlm_mask_num = (data["mlm_masks"]==1).sum().item()
            self.mlm_metric.update_loss(mlm_loss.item(),self.current_step)
            self.mlm_metric.update_acc(mlm_correct_num,mlm_mask_num,self.current_step)
            self.mlm_metric.update_lr(self.scheduler.get_last_lr()[0])

        if self.training_args.sop:
            sop_correct_num = result["so_pred"].argmax(dim=-1).eq(data["sop_labels"]).sum().item()
            self.sop_metric.update_loss(sop_loss.item(),self.current_step)
            self.sop_metric.update_acc(sop_correct_num,batch_size,self.current_step)
            self.sop_metric.update_lr(self.optim.param_groups[0]["lr"])


    
    def _print_metric(self):
        if self.training_args.mlm:
            self.mlm_metric.print_with_acc(self.current_step,self.total_steps)

        if self.training_args.sop:
            self.sop_metric.print_with_acc(self.current_step,self.total_steps)

    def _calculate_steps(self):

        # iteration을 끝까지 수행하여 epoch 당 step 수 측정
        step_num_in_pre_epoch = 0
        for _ in self.pre_train_data_loader:
            step_num_in_pre_epoch +=1
        self.total_steps = step_num_in_pre_epoch*self.pre_epoch

        step_num_in_post_epoch = 0
        for _ in self.post_train_data_loader:
            step_num_in_post_epoch+=1
        self.total_steps += step_num_in_pre_epoch*self.post_epoch
        self.warmup_steps=self.total_steps*self.training_args.warmup_ratio


    def _train_tokenizer(self,train_path,output_dir,vocab_size=32000, min_freq=3):
        """
            train_path의 학습데이터를 활용하여 Wordpice 학습 및 Vocab 생성
                - Special Token ([PAD], [MASK], [CLS]....)들 추가
        """

        tokenizer = BertWordPieceTokenizer(
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

    

    