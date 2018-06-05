TRAIN_PATH='data/finished_files/chunked/train_*'
VAL_PATH='data/finished_files/chunked/val_*'
TEST_PATH='data/finished_files/chunked/test_*'
VOCAB_PATH='data/finished_files/vocab'
EXP_NAME='exp_sample'
MAX_ITER=80000
SAVE_MODEL_EVERY=1000
MAX_TO_KEEP=5

BATCH_SIZE=8
MAX_ART_LEN=50
MAX_ENC_STEPS=600
LR=0.01
SELECTOR_LOSS_WT=5.0
INCONSISTENT_LOSS=True
INCONSISTENT_TOPK=3
SELECTOR_PATH='log/selector/exp_sample/eval/bestmodel-xxxx'
REWRITER_PATH='log/rewriter/exp_sample/train/model.ckpt_cov-81000'

# for eval mode
EVAL_METHOD='rouge'
DECODE_METHOD='greedy'
START_EVAL=1000
SINGLE_PASS=True  # if evaluating by loss, change singel_pass to False

# for evalall mode
SAVE_PKL=True   # save the results in pickle files
SAVE_VIS=False  # save for visualization (this data is big)
LOAD_BEST_EVAL_MODEL=False                                                                                                   
CKPT_PATH="log/end2end/$EXP_NAME/eval_rouge/bestmodel-xxxx"


#################
MODE='train'
#################


if [ "$MODE" = "train" ]
then
  python main.py --model=end2end --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=100 --max_train_iter=$MAX_ITER --batch_size=$BATCH_SIZE --max_art_len=$MAX_ART_LEN --lr=$LR --selector_loss_wt=$SELECTOR_LOSS_WT --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP --coverage=True --pretrained_selector_path=$SELECTOR_PATH --pretrained_rewriter_path=$REWRITER_PATH
elif [ "$MODE" = "eval" ]
then
  python main.py --model=end2end --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=120 --max_art_len=$MAX_ART_LEN --batch_size=64 --selector_loss_wt=$SELECTOR_LOSS_WT --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --eval_method=$EVAL_METHOD --decode_method=$DECODE_METHOD --start_eval_rouge=$START_EVAL --save_model_every=$SAVE_MODEL_EVERY --single_pass=$SINGLE_PASS --coverage=True
elif [ "$MODE" = "evalall" ]
then
  python main.py --model=end2end --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=$MAX_ENC_STEPS --max_dec_steps=120 --max_art_len=$MAX_ART_LEN --decode_method=beam --coverage=True --single_pass=1 --save_pkl=$SAVE_PKL --save_vis=$SAVE_VIS --inconsistent_loss=$INCONSISTENT_LOSS --inconsistent_topk=$INCONSISTENT_TOPK --eval_ckpt_path=$CKPT_PATH --eval_method=$EVAL_METHOD --load_best_eval_model=$LOAD_BEST_EVAL_MODEL
fi
