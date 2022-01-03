
import model.embedding as embedding
import json
import torch 

from transformers import BertTokenizer

def train():
    pass
def eval():
    pass
if __name__=="__main__":
    with open("./config/model.json","r") as cfg_json:
        config = json.load(cfg_json)

    input_emb = embedding.InputEmb(vocab_num=30522,seg_num=2,max_seq_len=512,d_model=config["hidden_dim"])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer("[CLS] the man worked as a waiter. [SEP] i am") # https://huggingface.co/docs/transformers/v4.14.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__

    print(input_emb(torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])))

        
