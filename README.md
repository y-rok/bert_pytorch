# Introduction 

This is the implementation of pre-training [Bert](https://arxiv.org/abs/1810.04805) with pytorch. 
(Masked language model, Sentence order prediction)

I have implemented transformer encoder model, dataset preprocessing from scratch except for tokenization.
I trained a small model with 10,000 sentences of book corpus and saw its loss converges.
However, due to the lack of GPUs, I have not tried training large model such as BERT-base, BERT-large.


# Quick Start

## Prepare your corpus

Prepare a txt file with sentences across multiple documents. 
(document is split by "\n")

```
i wish i had a better answer to that question .
starlings , new york is not the place youd expect much to happen .
.....
its a small quiet town , the kind where everyone knows your name .

i flipped open the pad and wrote : walking home .
i shoved it back in my pocket and continued walking .
...
```

## Pre-train your bert model



..
