import logging

MASK_TOKEN="[MASK]"
VOCAB_FILE_NAME="vocab"
VOCAB_TXT_FILE_NAME="vocab.txt"
MODEL_FILE_NAME="model.pt"
TRAINING_LOG_NAME="train.log"
MODEL_CONFIG_NAME="model_config.json"
TRAINING_ARGS_NAME="training_args.json"

def get_logger():
    logging.basicConfig()
    logger = logging.getLogger("logger")
    return logger
    
def add_handler_to_logger(logger, file_path):
    handler = logging.FileHandler(file_path)
    logger.addHandler(handler)
    
logger = get_logger()

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

        self.writer.add_scalar(self.name+" Acc",self.acc,step_num)

    def update_lr(self,lr):
        self.lr=lr
    
    def print_with_acc(self,step,total_step,with_acc=True):
        if with_acc:
            logger.info("step=%d/%d loss = %.4f, acc %.2f learning_rate =%.8f "%(step,total_step,self.loss,self.acc,self.lr))
        else:
            logger.info("step=%d/%d loss = %.4f learning_rate =%.8f "%(step,total_step,self.loss,self.lr))


