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
import bert_dataset
from vocab import WordVocab
from torch.utils.data import DataLoader
from plm_dataset import PLMDataset
    

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

def google_prepare_data():
    """
        google의 pre-training data 만드는 방식 확인
    """
        
    
if __name__=="__main__":
    test_shape_encoder()
    # test_dataloader_old()
    # test_data_loader()
    # test_pos_emb()
