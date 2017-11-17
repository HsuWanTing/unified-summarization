DATA_PATH='cnn_daily_chunked/train_*'
TEST_PATH='cnn_daily_chunked/test_*'
VOCAB_PATH='cnn_daily_chunked/vocab'
EXP_NAME='sample_exp3'  # please change the experiment name to yours!!
TRAIN_METHOD='PG'
MAX_TO_KEEP=3
SAVE_MODEL_EVERY=10
MAX_ITER=40000
USE_BASELINE=True
COV_LOSS_W=0.1
LR=0.0015       # learning rate (maybe should be smaller)
REWARD_RATIO=1.5

python -u run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --training_method=$TRAIN_METHOD --coverage=True --max_train_iter=$MAX_ITER --save_model_every=$SAVE_MODEL_EVERY --model_max_to_keep=$MAX_TO_KEEP --use_baseline=$USE_BASELINE --cov_loss_wt=$COV_LOSS_W --lr=$LR --reward_ratio=$REWARD_RATIO
