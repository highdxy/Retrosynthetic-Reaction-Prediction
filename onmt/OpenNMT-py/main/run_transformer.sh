SAMPLENUM=40000
SEQLEN=450

#python ../preprocess.py -train_src kescidata/train_sources_$SAMPLENUM -train_tgt kescidata/train_targets_$SAMPLENUM -valid_src kescidata/valid_a_sources_$SAMPLENUM -valid_tgt kescidata/valid_a_targets_$SAMPLENUM -src_vocab kescidata/vocab -tgt_vocab kescidata/vocab -save_data data/kesci -src_seq_length $SEQLEN -tgt_seq_length $SEQLEN


NUM=0
TRAIN=./train/$NUM
if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

python ../train.py \
       	-data data/kesci \
	-train_steps 1000000 \
       	-valid_steps 1000 \
	-save_checkpoint_steps 1000 \
	-save_model $TRAIN/kesci \
	-layers 4 \
	-rnn_size 128 \
	-word_vec_size 128 \
	-max_grad_norm 0 \
	-optim adam \
	-batch_size 16 \
	-encoder_type transformer \
	-decoder_type transformer \
       	-dropout 0.2 \
	-param_init 0 \
	-warmup_steps 2000 \
	-position_encoding \
	-learning_rate 0.05 \
	-gpu_ranks 0 -world_size 1 \
	-tensorboard \
	-tensorboard_log_dir $TRAIN

#STEP=1000000
#python ../translate.py -model $TRAIN/kesci_step_$STEP.pt -src kescidata/test_sources -output $TRAIN/test_targets_$STEP  -batch_size 1000 -gpu 0 -replace_unk -verbose
