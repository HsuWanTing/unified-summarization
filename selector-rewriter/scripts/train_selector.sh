DATA_PATH='data/CNN_Dailymail/add_extract_thres0.8_max400/chunked/train_*'
TEST_PATH='data/CNN_Dailymail/add_extract_thres0.8_max400/chunked/test_*'
VOCAB_PATH='data/CNN_Dailymail/add_extract_thres0.8_max400/vocab'
EXP_NAME='exp_test'
VOCAB_SIZE=50000
MAX_ITER=100
BATCH_SIZE=64

# 1-10000
python main.py --model=selector --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=100 --max_sent_len=50 --max_train_iter=$MAX_ITER --vocab_size=$VOCAB_SIZE --batch_size=$BATCH_SIZE


MAX_SELECT=35
THRES=0.1
EVAL_ROUGE=False
SAVE_PKL=False
PLOT=True


# 1-10000
python main.py --model=selector --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=100 --max_sent_len=50 --vocab_size=$VOCAB_SIZE --max_select_sent=$MAX_SELECT --single_pass=True --thres=$THRES --eval_rouge=$EVAL_ROUGE --save_pkl=$SAVE_PKL --plot=$PLOT #--load_best_val_model=True #--decode_ckpt_path=$DECODE_PATH
