
python -u main.py \
--mask_rate 0.6   \
--data_path ETTh1.csv  \
--enc_in 7  \
--dec_in 7 \
--train_percent 0.7 \
--train_epochs 5 \
--itr 1 \
--model_id ETTh1_0.6 \
--pre_train 960 \
--dropout 0.5 \
--learning_rate 0.01 \
--patch_len 16