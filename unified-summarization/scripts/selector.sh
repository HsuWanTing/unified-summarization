TRAIN_PATH='data/finished_files/chunked/train_*'
VAL_PATH='data/finished_files/chunked/val_*'
TEST_PATH='data/finished_files/chunked/test_*'
VOCAB_PATH='data/finished_files/vocab'
EXP_NAME='exp_sample'

# for train mode
MAX_ITER=50000
BATCH_SIZE=64
SAVE_MODEL_EVERY=10
MAX_TO_KEEP=3
#PRETRAINED=''  # uncomment this if you have pretrained selector model

# for evalall mode
SELECT_METHOD='prob'
MAX_SELECT=30
THRES=0.5
SAVE_PKL=True
LOAD_BEST_EVAL_MODEL=False
CKPT_PATH="log/selector/$EXP_NAME/eval/bestmodel-xxxx"

#################
MODE='train'
#################


if [ "$MODE" = "train" ]
then
  python main.py --model=selector --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --max_train_iter=$MAX_ITER --batch_size=$BATCH_SIZE --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP #--pretrained_selector_path=$PRETRAINED
elif [ "$MODE" = "eval" ]
then
  python main.py --model=selector --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --batch_size=$BATCH_SIZE
elif [ "$MODE" = "evalall" ]
then
  python main.py --model=selector --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_art_len=50 --max_sent_len=50 --max_select_sent=$MAX_SELECT --single_pass=True --select_method=$SELECT_METHOD --thres=$THRES --save_pkl=$SAVE_PK --eval_ckpt_path=$CKPT_PATH --load_best_eval_model=$LOAD_BEST_EVAL_MODEL
fi
