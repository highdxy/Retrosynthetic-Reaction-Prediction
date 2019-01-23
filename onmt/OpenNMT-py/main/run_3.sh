SAMPLENUM=40000
SEQLEN=450


#python ../preprocess.py -train_src kescidata/train_sources -train_tgt kescidata/train_targets -valid_src kescidata/valid_a_sources_$SAMPLENUM -valid_tgt kescidata/valid_a_targets_$SAMPLENUM -src_vocab kescidata/vocab -tgt_vocab kescidata/vocab -save_data data/kesci -src_seq_length $SEQLEN -tgt_seq_length $SEQLEN

#python ../preprocess.py -train_src kescidata/train_sources_$SAMPLENUM -train_tgt kescidata/train_targets_$SAMPLENUM -valid_src kescidata/valid_a_sources_$SAMPLENUM -valid_tgt kescidata/valid_a_targets_$SAMPLENUM -src_vocab kescidata/vocab -tgt_vocab kescidata/vocab -save_data data/kesci -src_seq_length $SEQLEN -tgt_seq_length $SEQLEN

NUM=3
TRAIN=./train/$NUM
if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

#python ../train.py \
#	-word_vec_size 128 \
#	-encoder_type rnn \
#	-decoder_type rnn \
#	-enc_layers 2 \
#	-dec_layers 2 \
#	-rnn_size 256 \
#	-global_attention general \
#	-data data/kesci \
#	-save_model $TRAIN/kesci \
#	-save_checkpoint_steps 1000 \
#	-gpu_ranks 0 -world_size 1 \
#	-optim adam \
#	-train_steps 800000 \
#       	-valid_steps 1000 \
#	-learning_rate 0.01 \
#	-tensorboard \
#	-tensorboard_log_dir $TRAIN
#

STEP=6000
python ../translate.py -model $TRAIN/kesci_step_$STEP.pt -src kescidata/valid_b_sources_2000 -output $TRAIN/valid_b_targets_$STEP  -batch_size 100 -gpu 0 -replace_unk -verbose
