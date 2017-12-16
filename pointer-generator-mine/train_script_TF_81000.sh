DATA_PATH='cnn_daily_chunked/train_*'
TEST_PATH='cnn_daily_chunked/test_*'
VOCAB_PATH='cnn_daily_chunked/vocab'
EXP_NAME='CNNDM_TF_81000iter'
MAX_ITER=10000

# 1-10000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=50 --max_dec_steps=15 --max_train_iter=$MAX_ITER 
# 10001-20000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=100 --max_dec_steps=25 --max_train_iter=$MAX_ITER 
# 20001-30000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=150 --max_dec_steps=40 --max_train_iter=$MAX_ITER 
# 30001-40000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=200 --max_dec_steps=50 --max_train_iter=$MAX_ITER 
# 40001-50000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=250 --max_dec_steps=60 --max_train_iter=$MAX_ITER 
# 50001-60000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=300 --max_dec_steps=80 --max_train_iter=$MAX_ITER 
# 60001-70000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=350 --max_dec_steps=100 --max_train_iter=$MAX_ITER 
# 70001-80000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=$MAX_ITER

python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100   --coverage=True --convert_to_coverage_model=True
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=100 --max_train_iter=1000 --save_model_every=500 --coverage=True

# decode
python run_summarization.py --mode=decode --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_dec_steps=120 --coverage=True --single_pass=1


