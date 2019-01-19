import collections

from tqdm import tqdm
from rdkit.Chem import AllChem,Draw


def gpu_config(gpu_num):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def vis_rdkit(index, smiles,img_save_path='../results/visualization/'):
    mol = AllChem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, img_save_path+str( index) +'_'+smiles+'.png')

def vis_test(test_file='test_targets_40000',test_path='../results/'):
    with open(test_path+test_file, 'r') as f:
        data = f.readlines()
    for idx, row in enumerate(data):
        smiles = row.strip().split('.')
        for smile in smiles:
            smile = smile.replace(' ','')
            try:
                vis_rdkit(idx,smile)
            except ValueError as e:
                print('ValueError', e)
                continue
    print('Done.')

def compute_exact(r_gold, r_pred):
    return int(r_gold == r_pred)

def compute_f1(r_gold, r_pred):
    gold_toks = r_gold.split(' ') 
    pred_toks = r_pred.split(' ')
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision   = 1.0 * num_same / len(pred_toks)
    recall      = 1.0 * num_same / len(gold_toks)
    f1          = (2 * precision * recall) / (precision + recall)
    return f1

def eval(test_gold, test_pred):
    with open(test_gold, 'r') as fgold:
        gold_lines = fgold.readlines()[:10]
    with open(test_pred, 'r') as fpred:
        pred_lines = fpred.readlines()

    assert(len(gold_lines) == len(pred_lines))
    
    test_num    = len(gold_lines)
    exact_num   = 0
    f1_all      = 0.0
    
    exact       = 0.0
    f1          = 0.0

    for i in tqdm( range(test_num) ):
        exact_num   += compute_exact(gold_lines[i], pred_lines[i])
        f1_all      += compute_f1(gold_lines[i], pred_lines[i]) 
    
    exact       = 1.0 * exact_num / test_num
    f1          = 1.0 * f1_all / test_num
    score       = 0.75 * f1 + 0.25 * exact
    
    return 'f1: '+str(f1), 'em:  '+str(exact), 'score:    '+str(score)

if __name__ =='__main__':
    
    print(eval('../rawdata/test_targets', '../train/0/test_targets_'))
    
    #vis_test()

    smiles0 = 'CCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4'
    smiles1 = 'CCCc1nc(C)c2C(=O)N=C(Nn12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(CC)CC4'
    #vis_rdkit(smiles1)
