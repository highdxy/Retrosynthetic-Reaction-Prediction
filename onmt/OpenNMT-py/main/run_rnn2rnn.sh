NUM=1
TRAIN=./train/$NUM
if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

python ../train.py \
       	-data data/kesci \
	-train_steps 800000 \
       	-valid_steps 1000 \
	-save_checkpoint_steps 1000 \
	-save_model $TRAIN/kesci \
	-rnn_size 128 \
	-word_vec_size 128 \
	-max_grad_norm 0 \
	-optim adam \
	-encoder_type rnn \
	-decoder_type rnn \
	-enc_layers 1 \
	-dec_layers 1 \
	-global_attention general \
	-learning_rate 0.001 \
	-gpu_ranks 0 -world_size 1 \
	-tensorboard \
	-tensorboard_log_dir $TRAIN


#STEP=5000
#python ../translate.py -model $TRAIN/kesci_step_$STEP.pt -src kescidata/valid_b_sources -output $TRAIN/valid_b_targets_$STEP  -batch_size 500 -gpu 0 -replace_unk -verbose
