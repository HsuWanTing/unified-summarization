DATA_PATH='dailymail_image_data/finished_files/chunked/train_*'
#DATA_PATH='/media/HDD/VSLab-VL/cindy/Workspace/summarization/dataset/cnn_sideinfo/cnn/finished_files'
TEST_PATH='dailymail_image_data/finished_files/chunked/test_*'
VOCAB_PATH='dailymail_image_data/finished_files/vocab'
EXP_NAME='DM_TF_44000_with_cap_attention'
TRAIN_METHOD='TF'
MAX_ITER=5000

# 1-5000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=50 --max_cap_steps=40 --max_dec_steps=15 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 5001-10000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=100 --max_cap_steps=80 --max_dec_steps=25 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 10001-15000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=150 --max_cap_steps=100 --max_dec_steps=40 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 15001-20000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=200 --max_cap_steps=120 --max_dec_steps=50 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 20001-25000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=250 --max_cap_steps=140 --max_dec_steps=60 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 25001-30000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=300 --max_cap_steps=160 --max_dec_steps=80 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 30001-35000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=350 --max_cap_steps=180 --max_dec_steps=100 --max_train_iter=$MAX_ITER --training_method=$TRAIN_METHOD
# 35001-43000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_cap_steps=200 --max_dec_steps=100 --max_train_iter=8000 --training_method=$TRAIN_METHOD

python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_cap_steps=200 --max_dec_steps=100 --training_method=$TRAIN_METHOD --coverage=True --convert_to_coverage_model=True
# 43000-44000
python run_summarization.py --mode=train --data_path=$DATA_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_cap_steps=200 --max_dec_steps=100 --max_train_iter=1000 --training_method=$TRAIN_METHOD --save_model_every=500 --coverage=True

# decode
python run_summarization.py --mode=decode --data_path=$TEST_PATH --vocab_path=$VOCAB_PATH --log_root=log --exp_name=$EXP_NAME --max_enc_steps=400 --max_cap_steps=200 --max_dec_steps=120 --coverage=True --single_pass=1
