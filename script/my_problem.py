# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems

from utils import util
util.gpu_config(0)

@registry.register_problem
class MyProblem(text_problems.Text2TextProblem):
    
    @property
    def vocab_type(self):
        return 'tokens'
    @property
    def oov_token(self):
       return None
    @property
    def vocab_filename(self):
        return 'vocab'

    #@property
    #def approx_vocab_size(self):
    #    return 2**11
    #@property
    #def max_subtoken_length(self):
    #    return 10
    
    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split
        
        import pandas as pd
        data = pd.read_csv('rawdata/data',header=None,sep='\t',error_bad_lines=False)
        train_num = data.shape[0]
        #import pdb
        #pdb.set_trace()
        for i in range(train_num):
            en = data.iloc[i][0]
            zh = data.iloc[i][1]
            if pd.isnull(en) or pd.isnull(zh):
                continue
            yield {
                "inputs": en,
                "targets": zh
            }

    def generate_samples_(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        q_r = open("./rawdata/q.txt", "r")
        a_r = open("./rawdata/a.txt", "r")

        comment_list = q_r.readlines()
        tag_list = a_r.readlines()
        q_r.close()
        a_r.close()
        for comment, tag in zip(comment_list, tag_list):
            comment = comment.strip()
            tag = tag.strip()
            # print comment, tag
            yield {
                "inputs": comment,
                "targets": tag
            }
