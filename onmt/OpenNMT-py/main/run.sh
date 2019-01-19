#python ../preprocess.py -train_src rawdata/train_sources -train_tgt rawdata/train_targets -valid_src rawdata/valid_sources -valid_tgt rawdata/valid_targets -src_vocab rawdata/vocab -tgt_vocab rawdata/vocab -save_data data/rs

NUM=0
TRAIN=./train/$NUM
if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

python ../train.py \
       	-data data/rs \
	-train_steps 200000 \
       	-valid_steps 100 \
	-save_checkpoint_steps 100 \
	-save_model $TRAIN/rs \
	-layers 4 \
	-rnn_size 128 \
	-word_vec_size 128 \
	-max_grad_norm 0 \
	-optim adam \
	-encoder_type transformer \
	-decoder_type transformer \
       	-dropout 0.2 \
	-param_init 0 \
	-warmup_steps 2000 \
	-position_encoding \
	-learning_rate 0.05 \
	-decay_method noam \
	-gpu_ranks 0 -world_size 1 \
	-tensorboard \
	-tensorboard_log_dir $TRAIN

#python ../translate.py -model model/rs_step_120000.pt -src rawdata/test_sources -output results/test_targets_  -batch_size 1000 -gpu 0 -replace_unk -verbose

