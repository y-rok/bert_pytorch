
# deubgging (간단한 데이터, 작은 모델로 모델 수렴 테스트)
TOKENIZERS_PARALLELISM=false

# 128 max seq len으로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_without_ft --ft_ratio 0 --mlm
# 처음 90% max seq len 64로 이후 128로 학습
# python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_ft --ft_seq_len 64 --mlm
# 처음 70% max seq len 64로 이후 128로 학습
python plm_train.py --config_path /root/data/ojt/config/bert_debug.json --train_path /root/data/ojt/datasets/book_corpus_debug.txt --output_dir /root/data/ojt/output/debug_model_mlm_ft_7_3 --ft_seq_len 64 --ft_ratio 0.7 --mlm