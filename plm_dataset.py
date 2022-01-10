

from torch.utils.data import IterableDataset
import random
import utils 
from itertools import chain
import torch
import pickle
import os 

logger = utils.get_logger()


class PLMDataset(IterableDataset):
    """
        corpus를 읽어서 Pretraining Language Model 학습 데이터를 준비
            - Masked Language Modeling
            - Sentence Order Prediction
    """
    def __init__(self,data_path,tokenizer,data_max_seq_len,model_max_seq_len,max_mask_tokens=20,cached_dir=None,use_cache=True,mlm_data=False,sop_data=False,seed=10) -> None:
        """
            - Corpus를 읽고 Tokenization 수행하여 Documents 객체를 pkl 파일로 저장
            - 이미 저장한 pkl 파일이 있다면 (Corpus를 다시 읽지 않고)해당 파일을 읽어서 사용 
        """
        self._set_data_max_seq_len(data_max_seq_len)
        self.tokenizer=tokenizer
        self.model_max_seq_len=model_max_seq_len
        self.max_mask_tokens=max_mask_tokens
        self.vocab=self.tokenizer.get_vocab()
        self.id_to_vocab={v:k for k,v in self.vocab.items()}
        self.sop_data=sop_data
        self.mlm_data=mlm_data

        random.seed(seed)
        
        self.doc_index = 0 
        self.sent_index = 0
        self.documents=[[]] # 3d token list - [doucments, sentences, tokens]

        dir_path = os.path.dirname(data_path)
        file_name,extension = os.path.splitext(os.path.basename(data_path))
        if cached_dir is None:
            cached_dir=dir_path

        pkl_file_path = os.path.join(cached_dir,file_name+".pkl")
        if os.path.exists(pkl_file_path) and use_cache:
            logger.info("Reading pkl file = %s"%pkl_file_path)
            with open(pkl_file_path,"rb") as f:
                self.documents=pickle.load(f)
        else:
            logger.info("Start Loading training data from %s"%data_path)
            with open(data_path,"r") as f:
                lines=f.readlines()

                for i,line in enumerate(lines):
                    if line=="\n":
                        self.documents.append([])
                        continue
                        
                    self.documents[-1].append(tokenizer.tokenize(line))
                    if i%100000==0:
                        logger.info("loading %d/%d"%(i,len(lines)))
                

            logger.info("Training data successfully loaded")
            logger.info("Writing pkl file = %s"%pkl_file_path)
            with open(pkl_file_path,"wb") as f:
                pickle.dump(self.documents,f)
            logger.info("Successfully written")

        sent_num = 0
        for doc in self.documents:
            sent_num+=len(doc)
        logger.info("doc num = %d, sentence num = %d"%(len(self.documents),sent_num))

    def _set_data_max_seq_len(self, data_max_seq_len):
        self.data_max_seq_len=data_max_seq_len
        self.max_token_num = data_max_seq_len-3 # [CLS], (segment 사이의) [SEP], (문장 마지막) [SEP] 제외한 최대 Token 수
    def __iter__(self):
        
        while True:

            # max_seq_len을 넘지 않는 sequence 생성 
            # sop 학습할 경우 - 임의로 Sentence order swapping
            is_end, sop_label, input_tokens, seg_a_token_num = self._get_sequence()
 
            # 더이상 불러올 데이터가 없는 경우 
            if is_end and sop_label==None:
                break
            else: 
                data={}
                if self.mlm_data:
                    # MLM 적용
                    input_tokens, mlm_labels, mlm_positions, mlm_masks = self._get_mlm_sequence(input_tokens, seg_a_token_num)
                    
                # Tensor 생성
                result=self.tokenizer([input_tokens[:seg_a_token_num]],[input_tokens[seg_a_token_num:]],is_split_into_wrods=True,
                max_length=self.model_max_seq_len,padding="max_length",return_token_type_ids=True)
                
                # [max_seq_len]
                data["input_ids"] = torch.tensor(result["input_ids"][0],dtype=torch.int) 
                data["seg_ids"]=torch.tensor(result["token_type_ids"][0],dtype=torch.int)
                data["att_masks"]=torch.tensor(result["attention_mask"][0],dtype=torch.int)
                # [1]
                data["sop_labels"]=torch.tensor(sop_label,dtype=torch.long)


                if self.mlm_data:
                    # [max_token_num]
                    data["mlm_labels"]=torch.tensor(mlm_labels,dtype=torch.long)
                    data["mlm_positions"]=torch.tensor(mlm_positions,dtype=torch.int)
                    data["mlm_masks"]=torch.tensor(mlm_masks,dtype=torch.int)

                # only sop
                elif self.sop_data:
                    
                    data["mlm_labels"]=torch.tensor([self.max_mask_tokens],dtype=torch.long)
                    data["mlm_positions"]=torch.tensor([self.max_mask_tokens],dtype=torch.int)
                    data["mlm_masks"]=torch.tensor([self.max_mask_tokens],dtype=torch.int)

                yield data

                if is_end:
                    break

    def _get_mlm_sequence(self,input_tokens, seg_a_token_num):
        """
            Masked Language Modeling 학습을 위해 token의 일부를 Masking
                - 전체 Token의 15% 혹은 최대 max_mask_tokens 수 만큼을 [MASK] token으로 바꿈
                    - 80%는 Masking
                    - 10%는 임의로 다른 vocab
                    - 10%는 유지 
        """
        
        masking_num = int(len(input_tokens)*0.15)
        if masking_num>self.max_mask_tokens: masking_num=self.max_mask_tokens

        mask_index_list = sorted(random.sample([i for i in range(len(input_tokens))],masking_num))
        # masked_tokens=[]


        mlm_labels=[]
        mlm_positions=[]
        mlm_masks=[]

        for i in mask_index_list:
            token=input_tokens[i]
            
            mlm_labels.append(self.vocab[token])

            if i<seg_a_token_num:
                mlm_positions.append(i+1) # +1 / [CLS]
            else:
                mlm_positions.append(i+2) # +2 / [CLS], [SEP]

            mlm_masks.append(1)

            prob=random.randint(0,99)
    

            if prob<80:
                # Masking
                input_tokens[i]=utils.MASK_TOKEN
                
            elif prob<90:
                # 다른 vocab으로 취환
                while True:
                    rand_vocab_index = random.randint(0,len(self.vocab)-1)
                    rand_vocab=self.id_to_vocab[rand_vocab_index]

                    # [PAD], [MASK], [CLS], [SEP], [UNUSED] 등의 token을 뽑았을 경우 다시 뽑음
                    if not(rand_vocab.startswith("[") and rand_vocab.endswith("]")):
                        break
                
                input_tokens[i]=rand_vocab
            else:
                # 유지
                continue
            # else:
            #     labels.append(0)

        # mask token의 수가 max_mask_tokens 보다 적은 경우 padding 
        if len(mlm_labels)<self.max_mask_tokens:
            num_pad = (self.max_mask_tokens-len(mlm_labels))
            mlm_labels.extend([0]*num_pad)
            mlm_positions.extend([0]*num_pad)
            mlm_masks.extend([0]*num_pad)
            
        return input_tokens, mlm_labels,mlm_positions, mlm_masks


    def _get_sequence(self):
        """
            SOP를 고려하여 2개의 Segment로 이루어진 Sequence 셍성
                - Segment A, B를 구성하는 Sequence 길이는 임의로 결정됨
                    - 각 Segment의 Sequence는 연속된 Sentence들로 구성됨 (sentence의 일부 token만 segment에 포함되는 경우는 없음.)
                    - 전체 Sequence는 token 수가 max_seq_len을 넘지 않는선에 가능한 많은 Sentence들로 구성됨
                - 50%의 확률로 Sentence Order의 Swap 여부 결정 ( * BERT의 NSP 대신 ALBERT의 Sentence Order Prediction 사용 )
                    - True -> Segment A, B는 순서를 유지하며 연결되는 Sequence
                    - False -> Segment A, B를 구성하는 Sequence를 서로 바꿈      
        """

        sequence=[] # sentence lists
        seq_len=0 

        is_end = False
        next_is_new_doc=False

        

        """ sequence에 들어갈 Sentence list 생성 """
        while True:

            def __next_index():
                is_new_doc = True

                # last sentence
                if len(self.documents[self.doc_index])-1==self.sent_index:
                    self.doc_index=self.doc_index+1
                    self.sent_index=0
                    return is_new_doc
                else:
                    self.sent_index+=1
                    is_new_doc = False
                    return is_new_doc
            
            """ 추가할 문장에 대한 확인 """
            # 새로운 문서의 Sentence를 다루는 경우
            if next_is_new_doc:
                # 이전까지 생성된 Sequence만을 활용
                if self.doc_index==len(self.documents):
                    if len(sequence)<=1:
                        return True, None, None, None
                    else:
                        is_end=True
                        break
                else:
                    break
                
            """ 문장을 sequence에 추가 """
            new_sentence = self.documents[self.doc_index][self.sent_index]
            
            # sentence 추가 시 max_seq_len 넘는 경우
            if seq_len+len(new_sentence)>self.max_token_num:
                # max_seq_len 보다 긴 첫번째 문장은 skip
                if seq_len==0:
                    # logger.warning(new_sentence)
                    logger.debug("skip sentence longer than max_seq_len-3 (sentence lenth = "+str(len(new_sentence))+")")
                # 연속 두문장의 길이가 max_seq_len 보다 길 때 -> 
                elif len(sequence)==1:
                    # 1번쨰 문장 버리고 2번째 문장 추가
                    if len(new_sentence)<self.max_mask_tokens:
                        seq_len=len(new_sentence)
                        sequence=[new_sentence]
                    else:
                        seq_len=0
                        sequence=[]
                    # __add_new_sentence()
                else:
                    break
            else:
                sequence.append(new_sentence)
                seq_len+=len(new_sentence)
            
            """ 문장 추가 후 새로 추가할 문장에 대한 index 갱신"""
            next_is_new_doc = __next_index()

        
        """ 생성한 sequence에 sop 적용하여 데이터 생성 """
        input_tokens=[]

        if self.sop_data:
            # 50% 확률로 Sentence 바꿀지 여부 결정
            correct_order = 1 if random.randint(0,1)==0 else 0
        else:
            correct_order = 1

        # 임의로 Segment A의 문장 수를 정함
        seg_a_num = random.randint(1,len(sequence)-1)
        
        if correct_order==0:
            seg_b_tokens=list(chain(*sequence[0:seg_a_num]))
            seg_a_tokens=list(chain(*sequence[seg_a_num:]))
            input_tokens=seg_a_tokens+seg_b_tokens

            seg_a_token_num = len(seg_a_tokens)

        else:
            input_tokens=list(chain(*sequence))
            seg_a_token_num = len(list(chain(*sequence[0:seg_a_num])))
            
        
        return is_end, correct_order, input_tokens, seg_a_token_num
        



