# # from transformers import BertTokenizer, BertModel
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained("bert-base-uncased")
# # text = "Replace me by any text you'd like."
# # encoded_input = tokenizer(text, return_tensors='pt')
# # output = model(**encoded_input)

# # print(output)
import unittest, math
import torch
from model import embedding
from model import encoder
import json
# import torch
from transformers import BertTokenizer
# import bert_dataset
# from vocab import WordVocab
from torch.utils.data import DataLoader
from plm_dataset import PLMDataset
from model.bert import Bert
import sys
import utils
# from pre_train import train_tokenizer
import os

# def test_pos_emb_init():
#     """
#         positional embedding 초기화에 대해 Test
#     """
#     max_seq_len=4
#     emb_dim=2

#     actual = torch.zeros((max_seq_len,emb_dim))


#     def _get_pos_emb(pos_index, dim_index):
#         if dim_index%2==0:
#             return math.sin(pos_index/math.pow(10000,(dim_index)/emb_dim))
#         else:
#             return math.cos(pos_index/math.pow(10000,(dim_index-1)/emb_dim))

#     for i in range(max_seq_len):
#         for j in range(emb_dim):
#             actual[i,j]=_get_pos_emb(i,j)

#     pos_emb = embedding.PositionEnc(max_seq_len=max_seq_len,d_model=emb_dim)
#     expected = pos_emb()

#     # print(actual)
#     # print(expected)
#     torch.testing.assert_close(actual,expected)

# def test_att_no_padd():
#     pass

# def test_att_with_padd():
#     pass


# def get_input_tensor(inputs,max_seq_len):
#     """
#         Tokenizer의 output으로부터 encdoer의 input tensor 생성
        
#         Args:
#             inputs (dic): {"input_ids":[2d list],"token_type_ids":[2d list],"attention_mask":[2d_list]}
#                 https://huggingface.co/docs/transformers/master/en/main_classes/tokenizer#transformers.PreTrainedTokenizer 참고
#         Return:
#             seq_len_list (list): input sequence length들의 list
#             input_ids (Tensor): max_seq_len만큼 0으로 padding input id tensor
#             seg_ids (Tensor): max_seq_len만큼 0으로 padding segment id tensor
#     """
#     batch_size=len(inputs["input_ids"])

#     def get_padded_tensor(list_2d):
#         x = torch.zeros((batch_size,max_seq_len),dtype=torch.int)
#         for index, ex in enumerate(list_2d):
#             x[index,:len(ex)]=torch.tensor(ex,dtype=torch.int)
#         return x
        
        
#     input_ids = get_padded_tensor(inputs["input_ids"])
#     seg_ids = get_padded_tensor(inputs["token_type_ids"])
#     seq_len_list = [len(x) for x in inputs["input_ids"]]
#     # att_mask = get_padded_tensor(inputs["attention_mask"])
    
#     return seq_len_list, input_ids, seg_ids
    
    

def test_shape_encoder():
    """ 
        Bert에 batch input을 넣었을 때 output의 shape 확인
    """
    
    with open("./config/model_debug.json","r") as cfg_json:
        config = json.load(cfg_json)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 0번재 vocab이 [pad]
    
    vocab_num = 30522
    
    # https://huggingface.co/docs/transformers/v4.14.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__
    inputs = tokenizer([["hello.","second sentence"],["second batch","test"]],max_length=config["max_seq_len"],padding="max_length",return_token_type_ids=True) 

    input_ids = torch.tensor(inputs["input_ids"],dtype=torch.int)
    seg_ids =torch.tensor(inputs["token_type_ids"],dtype=torch.int)
    masks=torch.tensor(inputs["attention_mask"],dtype=torch.int)

    # seq_len_list, input_ids, seg_ids = get_input_tensor(inputs,config["max_seq_len"])
    
    bert=encoder.Encoder(vocab_num,config["seg_num"],config["layer_num"],config["head_num"],config["max_seq_len"],config["d_model"],config["d_k"],config["d_ff"],config["dropout"])
    out=bert(input_ids,seg_ids,masks)
    
    assert out.size() == (2,config["max_seq_len"],config["d_model"]), print(out.size())
    
def test_data_loader():
    """
        dataloader를 활용하여 불러온 input data 확인
    """

    data_path = "./datasets/book_corpus_debug.txt"

    # debug 데이터셋 제작
    with open("./datasets/books_large_p1.txt","r") as f:
        data=[]
        for index in range(80):
            data.append(f.readline())
            if index!=0 and index %30 ==0:
                data.append("\n")

        with open(data_path,"w") as wf:
            wf.writelines(data)
    
    with open("./config/model_debug.json","r") as cfg_json:
        config = json.load(cfg_json)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    plm_dataset=PLMDataset(data_path,tokenizer,config["max_seq_len"])
    train_data_loader = DataLoader(plm_dataset, batch_size=2)

    iterator =iter(train_data_loader)

    print(next(iterator))
    print(next(iterator))
    
def test_dataloader_old():

    # debug 데이터셋 제작
    with open("./datasets/books_large_p1.txt","r") as f:
        data=[]
        for _ in range(50):
            data.append(f.readline().strip())

        with open("./datasets/book_corpus_debug.txt","w") as wf:
            for index in range(50):
                wf.write(data[index*2]+"\t"+data[index*2+1]+'\n')

    with open("./datasets/book_corpus_debug.txt","r") as f:
        vocab = WordVocab(f)

    with open("./config/model_debug.json","r") as cfg_json:
        config = json.load(cfg_json)

    
    # vocab = WordVocab.load_vocab("./datasets/vocab.txt")

    plm_dataset = bert_dataset.BERTDataset("./datasets/book_corpus_debug.txt",vocab,config["max_seq_len"])
    train_data_loader = DataLoader(plm_dataset, batch_size=2)

    print(next(iter(train_data_loader)))



def test_predict_mask_token():
    
    with open("/root/data/ojt/config/bert_debug.json","r") as cfg_json:
        config = json.load(cfg_json)

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    vocab=tokenizer.get_vocab()

    bert=Bert(config=config,tokenizer=tokenizer,with_cuda=False)
    # model_path = "/root/data/ojt/output/books_large_p1_25.pt"
    model_path = "/root/data/ojt/output/debug_model/checkpo.pt"

    print("Loading model %s"%model_path)
    bert.load_state_dict(torch.load(model_path))
    bert.eval()
    
    while True:
        text = sys.stdin.readline()
        predict_mask_token(bert,text)

def predict_mask_token(bert,text,with_cuda=False):
    result = bert.tokenizer(text, max_length=bert.config["max_seq_len"],padding="max_length",return_token_type_ids=True)
    
    data ={}

    data["input_ids"] = torch.tensor([result["input_ids"]],dtype=torch.int) 
    data["seg_ids"]=torch.tensor([result["token_type_ids"]],dtype=torch.int)
    data["att_masks"]=torch.tensor([result["attention_mask"]],dtype=torch.int)

    mlm_positions=[]
    mlm_masks=[]
    for index,id in enumerate(result["input_ids"]):
        if id ==bert.vocab[utils.MASK_TOKEN]:
            mlm_positions.append(index)
            mlm_masks.append(1)

    if len(mlm_positions)<bert.config["max_mask_tokens"]:
        pad_num = bert.config["max_mask_tokens"]-len(mlm_positions)
        mlm_positions.extend([0]*pad_num)
        mlm_masks.extend([0]*pad_num)
    
    # [max_token_num]
    data["mlm_positions"]=torch.tensor([mlm_positions],dtype=torch.int)
    data["mlm_masks"]=torch.tensor([mlm_masks],dtype=torch.int)

    if with_cuda==True:
        if with_cuda:
                for key, item in data.items():
                    data[key]=item.cuda()

    result = bert(data,return_sop=False,return_mlm=True)
    # print(result)
    print(bert.convert_mask_pred_to_token(result["mask_pred"],data["mlm_masks"]))
        

# def create_train_data_for_bertpytorch():

#     config_path = "/root/data/ojt/bert_debug.json"
#     output_dir = "/root/data/ojt/reference/BERT-pytorch/dataset"
#     train_path = "/root/data/ojt/datasets/book_corpus_debug.txt"

#     with open(config_path,"r") as cfg_json:
#         config = json.load(cfg_json)

    
#     if os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     train_tokenizer(train_path,output_dir)

#     vocab_file_path = os.path.join(output_dir,utils.VOCAB_TXT_FILE_NAME)
#     tokenizer=BertTokenizer(vocab_file=vocab_file_path,do_lower_case=True)

#     plm_dataset=PLMDataset(train_path,tokenizer,config["max_seq_len"],config["max_mask_tokens"],cached_dir=output_dir)

if __name__=="__main__":
    test_predict_mask_token()
    # test_shape_encoder()

    