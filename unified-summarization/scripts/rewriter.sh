TRAIN_PATH='data/finished_files/chunked/train_*'
VAL_PATH='data/finished_files/chunked/val_*'
TEST_PATH='data/finished_files/chunked/test_*'
VOCAB_PATH='data/finished_files/vocab'
EXP_NAME='exp_sample'
MAX_ITER=10000
SAVE_MODEL_EVERY=1000
MAX_TO_KEEP=5

# for eval mode
EVAL_METHOD='rouge'
DECODE_METHOD='greedy'
START_EVAL=8000
SINGLE_PASS=True  # if evaluating by loss, change singel_pass to False

# for evalall mode
LOAD_BEST_EVAL_MODEL=False
CKPT_PATH=''

#################
MODE='train'
#################


if [ "$MODE" = "train" ]
then
  # 1-10000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=50 --max_dec_steps=15 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 10001-20000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=100 --max_dec_steps=25 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 20001-30000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=150 --max_dec_steps=40 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 30001-40000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=200 --max_dec_steps=50 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 40001-50000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=250 --max_dec_steps=60 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 50001-60000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=300 --max_dec_steps=80 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 60001-70000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=350 --max_dec_steps=100 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # 70001-80000
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP
  # add coverage mechanism for 1000 iter
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --coverage=True --convert_to_coverage_model=True
  python main.py --model=rewriter --mode=train --data_path=$TRAIN_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=1000 --save_model_every=200 --coverage=True --model_max_to_keep=$MAX_TO_KEEP
elif [ "$MODE" = "eval" ]
then
  python main.py --model=rewriter --mode=eval --data_path=$VAL_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=120 --coverage=False --batch_size=64 --eval_method=$EVAL_METHOD --decode_method=$DECODE_METHOD --start_eval_rouge=$START_EVAL --save_model_every=$SAVE_MODEL_EVERY --single_pass=$SINGLE_PASS
elif [ "$MODE" = "evalall" ]
then
  # decode
  python main.py --model=rewriter --mode=evalall --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=120 --coverage=True --decode_method=beam --single_pass=1 --eval_method=$EVAL_METHOD --load_best_eval_model=$LOAD_BEST_EVAL_MODEL --eval_ckpt_path=$CKPT_PATH
fi
