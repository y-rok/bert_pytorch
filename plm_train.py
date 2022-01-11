import argparse

from transformers import utils
from plm_trainer import LMTrainer
import json
import logging
from utils import get_logger
import os 
logger=get_logger()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,type=str, help="model configuration file path")
    parser.add_argument('--train_path', required=True, type=str, help="training data file path")
    parser.add_argument('--output_dir',required=True,type=str,help="the output path where model, vocab file will be saved")
    parser.add_argument('--steps',default=100,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--lr",default=0.001,help="learning rate",type=float)
    parser.add_argument("--betas",default=(0.9,0.999))
    parser.add_argument("--weight_decay",default=0.01)
    # parser.add_argument("--warmup_steps",default=10000,type=int)
    parser.add_argument("--warmup_ratio",default=0,type=float)
    parser.add_argument("--ft_seq_len",default=128,type=int)
    parser.add_argument("--ft_ratio",default=0.9,type=float)
    parser.add_argument("--epochs",default=1000,type=int)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("-warmup_steps",default=10000)
    parser.add_argument("--num_workers",type=int,default=2)
    parser.add_argument("--mlm",action="store_true")
    parser.add_argument("--sop",action="store_true")
    parser.add_argument("--log_steps",type=int,default=500)
    parser.add_argument("--save_steps",type=int,default=1000)
 

    training_args=parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    if training_args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


    
    logger.info("Reading model config from %s"%training_args.config_path)
    with open(training_args.config_path,"r") as cfg_json:
        model_config = json.load(cfg_json)

    trainer=LMTrainer(training_args,model_config)
    trainer.train()