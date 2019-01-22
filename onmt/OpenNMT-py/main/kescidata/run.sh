SAMPLE_NUM=40000

head -n $SAMPLE_NUM train_sources > train_sources_$SAMPLE_NUM
head -n $SAMPLE_NUM train_targets > train_targets_$SAMPLE_NUM

head -n $SAMPLE_NUM valid_a_sources > valid_a_sources_$SAMPLE_NUM
head -n $SAMPLE_NUM valid_a_targets > valid_a_targets_$SAMPLE_NUM
