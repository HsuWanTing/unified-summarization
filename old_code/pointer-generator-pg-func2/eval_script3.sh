DATA_PATH='cnn_daily_chunked/train_*'
TEST_PATH='cnn_daily_chunked/test_*'
VOCAB_PATH='cnn_daily_chunked/vocab'
EXP_NAME='sample_exp3' # please change the experiment name to yours!!
TRAIN_METHOD='PG'
MAX_TO_KEEP=5
SAVE_MODEL_EVERY=10
MAX_ITER=80000
USE_BASELINE=True
COV_LOSS_W=0.1
LR=0.0015     # learning rate (maybe should be smaller)
REWARD_RATIO=1.5

######################################################################
# Make sure that the settings for eval are the same as train mode!   #
# including:                                                         #
# 1. EXP_NAME                                                        #
# 2. USE_BASELINE                                                    #
# 3. COV_LOSS_W                                                      #
# 4. LR                                                              #
# 5. REWARD_RATIO                                                    #
######################################################################



python -u run_summarization.py --mode=eval --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --training_method=$TRAIN_METHOD --coverage=True --max_train_iter=$MAX_ITER --save_model_every=$MAX_ITER --model_max_to_keep=$MAX_TO_KEEP --use_baseline=$USE_BASELINE --cov_loss_wt=$COV_LOSS_W --lr=$LR --batch_size=64 --reward_ratio=$REWARD_RATIO
