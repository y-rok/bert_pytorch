
# deubgging (간단한 데이터, 작은 모델로 모델 수렴 테스트)
TOKENIZERS_PARALLELISM=false

# mlm 학습 테스트
# 128 max seq len으로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_without_ft --ft_ratio 0 --mlm
# 처음 90% max seq len 64로 이후 128로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_ft --ft_seq_len 64 --mlm
# 처음 70% max seq len 64로 이후 128로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_ft_7_3 --ft_seq_len 64 --ft_ratio 0.7 --mlm

# sop 학습 테스트
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_sop --sop

# mlm + sop 학습 테스트 
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_sop_ft_7_3 --ft_seq_len 64 --ft_ratio 0.7 --sop --mlm

# 디버그 데이터 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/book_bert_64_debug_mlm --ft_ratio 1 --ft_seq_len 64 --mlm  --epochs 3000
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/book_bert_debug_mlm --ft_ratio 0 --mlm  --epochs 3000
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/book_bert_debug_mlm_ft_7_3 --ft_ratio 0.7 --ft_seq_len 64 --mlm --epochs 3000
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/book_bert_debug_mlm_sop --ft_ratio 0 --mlm --sop --epochs 3000 

#100개의 문장으로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_100.txt --output_dir /root/data/ojt/output/book_100_bert_debug_mlm --ft_ratio 0 --mlm
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_100.txt --output_dir /root/data/ojt/output/book_100_bert_debug_mlm_ft_7_3 --ft_ratio 0.7 --ft_seq_len 64 --mlm
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/dcata/ojt/datasets/book_corpus_100.txt --output_dir /root/data/ojt/output/book_100_bert_debug_mlm_sop --ft_ratio 0 --mlm --sop

# 1000개 문장으로 학습
# export CUDA_VISIBLE_DEVICES=0,1
# python plm_train.py --config_path /root/data/ojt/config/bert_mini.json --train_path /root/data/ojt/datasets/books_corpus_p1_1.txt --output_dir /root/data/ojt/output/bert_debug_book_1_mlm_ft --ft_seq_len 64 --ft_ratio 0.9 --mlm --epochs 1000 --batch_size 64 


# 10000개 문장으로 학습
python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_1.txt --output_dir /root/data/ojt/output/bert_small_book_1_mlm --ft_ratio 0 --mlm --epochs 2000 --batch_size 32 --warmup_steps 0

# export CUDA_VISIBLE_DEVICES=0,1
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/books_corpus_p1_10.txt --output_dir /root/data/ojt/output/bert_debug_book_10_mlm_ft --ft_seq_len 64 --ft_ratio 0.9 --mlm --epochs 10000 --batch_size 64 

# export CUDA_VISIBLE_DEVICES=0,1d
# python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_10.txt --output_dir /root/data/ojt/output/bert_debug_book_10_mlm_ft --ft_seq_len 64 --ft_ratio 0.9 --mlm --epochs 10000 --batch_size 64

# export CUDA_VISIBLE_DEVICES=0,1
# python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_1.txt --output_dir /root/data/ojt/output/bert_small_book_1_mlm_ft --ft_seq_len 128 --ft_ratio 0.9 --mlm --epochs 10000 --warmup_steps 10000 --batch_size 64

# export CUDA_VISIBLE_DEVICES=2,3
# python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_10.txt --output_dir /root/data/ojt/output/bert_small_book_10_mlm_ft --ft_seq_len 128 --ft_ratio 0.9 --mlm --epochs 10000 --warmup_steps 10000 --batch_size 64


# export CUDA_VISIBLE_DEVICES=0,1
# python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_1.txt --output_dir /root/data/ojt/output/bert_small_book_1_sop_ft --ft_seq_len 128 --ft_ratio 0.9 --sop --epochs 10000 --warmup_steps 10000 --batch_size 64

# export CUDA_VISIBLE_DEVICES=2,3
# python plm_train.py --config_path /root/data/ojt/config/bert_small.json --train_path /root/data/ojt/datasets/books_corpus_p1_10.txt --output_dir /root/data/ojt/output/bert_small_book_10_sop_ft --ft_seq_len 128 --ft_ratio 0.9 --sop --epochs 10000 --warmup_steps 10000 --batch_size 64
