#t2t-datagen --t2t_usr_dir=script --problem=my_problem --data_dir=./data

NUM=0
TRAIN=./train/$NUM

if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

t2t-trainer --t2t_usr_dir=script --problem=my_problem --data_dir=./data --model=transformer --hparams_set=transformer_base_single_gpu --train_steps=50000 --output_dir=$TRAIN

#t2t-avg-all --model_dir=train/1/ --output_dir=train/merge_all/ 

#t2t-decoder --t2t_usr_dir=script --problem=my_problem --data_dir=./data --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=$TRAIN --decode_hparams="batch_size=5,beam_size=5,alpha=0.2" --decode_from_file=rawdata/test_sources_10 --decode_to_file=results/test_targets_40000_0.2
