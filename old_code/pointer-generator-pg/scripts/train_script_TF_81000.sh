DATA_PATH='cnn_daily_chunked/train_*'
TEST_PATH='cnn_daily_chunked/test_*'
VOCAB_PATH='cnn_daily_chunked/vocab'
EXP_NAME='CNNDM_TF_81000iter'
TRAIN_METHOD='TF'
MAX_ITER=10000

# 1-10000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=50 --max_dec_steps=15 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 10001-20000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=100 --max_dec_steps=25 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 20001-30000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=150 --max_dec_steps=40 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 30001-40000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=200 --max_dec_steps=50 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 40001-50000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=250 --max_dec_steps=60 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 50001-60000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=300 --max_dec_steps=80 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 60001-70000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=350 --max_dec_steps=100 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 70001-80000
python -u run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD 2>&1 | tee log_cnndm_400_100_81000

python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD --coverage=True --convert_to_coverage_model=True
python -u run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=1000 --training_method=$TRAIN_METHOD --save_model_every=500 --coverage=True 2>&1 | tee log_cnndm_400_100_cov_81000

# decode
python run_summarization.py --mode=decode --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=120 --coverage=True --single_pass=1

# PG training
#python -u run_summarization.py --mode=train --data_path=cnn_daily_chunked/train_* --vocab_path=cnn_daily_chunked/vocab --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --training_method=PG --coverage=True --max_train_iter=5000 --save_model_every=500

