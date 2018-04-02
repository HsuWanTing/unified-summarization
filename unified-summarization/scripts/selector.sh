TRAIN_PATH='data/CNN_Dailymail/finished_files/chunked/train_*'
VAL_PATH='data/CNN_Dailymail/finished_files/chunked/val_*'
TEST_PATH='data/CNN_Dailymail/finished_files/chunked/test_*'
VOCAB_PATH='data/CNN_Dailymail/finished_files/vocab'
EXP_NAME='xxxx'

# for train mode
MAX_ITER=100000
BATCH_SIZE=64
LOSS='CE'
SAVE_MODEL_EVERY=10
MAX_TO_KEEP=3
#PRETRAINED=''  # uncomment this if have pretrained selector model

# for policy gradient
RATIO=0.3
RATIO_WT=1.0

# for evalall mode
SELECT_METHOD='prob'
MAX_SELECT=30
THRES=0.5
EVAL_ROUGE=True
SAVE_PKL=True
SAVE_BIN=False
PLOT=True
EVAL_PATH="log/selector/$EXP_NAME/eval_val/bestmodel-xxxx"

#################
MODE='train'
#################


if [ "$MODE" = "train" ]
then
  python main.py --model=selector --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --max_train_iter=$MAX_ITER --rnn_type=GRU --batch_size=$BATCH_SIZE --loss=$LOSS --regu_ratio=$RATIO --regu_ratio_wt=$RATIO_WT --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP #--pretrained_selector_path=$PRETRAINED
elif [ "$MODE" = "eval" ]
then
  python main.py --model=selector --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --rnn_type=GRU --loss=$LOSS --batch_size=$BATCH_SIZE --regu_ratio=$RATIO --regu_ratio_wt=$RATIO_WT
elif [ "$MODE" = "evalall" ]
then
  python main.py --model=selector --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --rnn_type=GRU --max_select_sent=$MAX_SELECT --single_pass=True --select_method=$SELECT_METHOD --thres=$THRES --eval_rouge=$EVAL_ROUGE --save_pkl=$SAVE_PKL --save_bin=$SAVE_BIN --plot=$PLOT --load_best_val_model=True #--eval_ckpt_path=$EVAL_PATH
fi
