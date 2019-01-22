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

def compute_exact_(r_gold, r_pred):
    return r_gold.split('.') == r_pred.split('.')

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

def compute_f1_(r_gold, r_pred):
    gold_toks = r_gold.split('.')
    pred_toks = r_pred.split('.')
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision   = 1.0 * num_same / len(pred_toks)
    recall      = 1.0 * num_same / len(gold_toks)
    f1          = (2 * precision * recall) / (precision + recall)
    return f1



def data_split(space=True,datapath='../kescidata/',savepath='../kescidata/'):
    
    with open(datapath+'train.txt', 'r') as ftrain:
        data = ftrain.readlines()
        del(data[0])
    with open(datapath+'test.txt', 'r') as ftest:
        test = ftest.readlines()
        del(test[0])

    train_src = open(savepath+'train_sources', 'w+')
    train_tgt = open(savepath+'train_targets', 'w+')
    
    valid_a_src = open(savepath+'valid_a_sources', 'w+')
    valid_a_tgt = open(savepath+'valid_a_targets', 'w+')
    valid_b_src = open(savepath+'valid_b_sources', 'w+')
    valid_b_tgt = open(savepath+'valid_b_targets', 'w+')
    
    test_src  = open(savepath+'test_sources', 'w+')
    vocab_file= open(datapath+'vocab','w+') 
    
    import random
    random.seed(1)
    
    random.shuffle(data)
    
    kfolds = 5
    valid_b_num = 20000

    vocab = set()

    src_len = []
    tgt_len = []
    for i in tqdm ( range(0, int( len(data) * (1.0 - 1.0/5) ) ) ):   
    #for i in tqdm ( range(0, 10) ):   
        row = data[i][:data[i].find('|')]
        reac_list=row.split(',')[1].split('>')[0]
        reag_list=row.strip().split(',')[1].split('>')[1]
        prod_list=row.strip().split(',')[1].split('>')[2]
        
        if space:
            prod_list   = ' '.join(prod_list)
            reag_list   = ' '.join(reag_list)
            reac_list   = ' '.join(reac_list)
        
        vocab = set(reac_list) | set(reag_list) | set(prod_list) | vocab
        
        if space:
            src_len.append(len( prod_list+' . '+reag_list+'\n' ))
        else:
            src_len.append(len( prod_list+'.'+reag_list+'\n' ))
        tgt_len.append(len( reac_list+'\n' ))
        
        if space:
            train_src.write(prod_list+' . '+reag_list+'\n')
        else:
            train_src.write(prod_list+'.'+reag_list+'\n')
        train_tgt.write(reac_list+'\n')
    

    for j in tqdm ( range(int( len(data) * (1.0 - 1.0/5) ), len(data)-valid_b_num)):
    #for j in tqdm ( range(10, 14) ):   
        row = data[j][:data[j].find('|')]
        reac_list=row.split(',')[1].split('>')[0]
        reag_list=row.strip().split(',')[1].split('>')[1]
        prod_list=row.strip().split(',')[1].split('>')[2]
        
        if space:
            prod_list   = ' '.join(prod_list)
            reag_list   = ' '.join(reag_list)
            reac_list   = ' '.join(reac_list)
        
        vocab = set(reac_list) | set(reag_list) | set(prod_list) | vocab
        
        if space:
            src_len.append(len( prod_list+' . '+reag_list+'\n' ))
        else:
            src_len.append(len( prod_list+'.'+reag_list+'\n' ))
        tgt_len.append(len( reac_list+'\n' ))
        
        if space:
            valid_a_src.write(prod_list+' . '+reag_list+'\n')
        else:
            valid_a_src.write(prod_list+'.'+reag_list+'\n')
        valid_a_tgt.write(reac_list+'\n')
    
    for p in tqdm ( range(len(data)-valid_b_num, len(data))):
    #for j in tqdm ( range(10, 14) ):   
        row = data[p][:data[p].find('|')]
        reac_list=row.split(',')[1].split('>')[0]
        reag_list=row.strip().split(',')[1].split('>')[1]
        prod_list=row.strip().split(',')[1].split('>')[2]
        
        if space:
            prod_list   = ' '.join(prod_list)
            reag_list   = ' '.join(reag_list)
            reac_list   = ' '.join(reac_list)
        
        vocab = set(reac_list) | set(reag_list) | set(prod_list) | vocab
        
        if space:
            src_len.append(len( prod_list+' . '+reag_list+'\n' ))
        else:
            src_len.append(len( prod_list+'.'+reag_list+'\n' ))
        tgt_len.append(len( reac_list+'\n' ))
        
        if space:
            valid_b_src.write(prod_list+' . '+reag_list+'\n')
        else:
            valid_b_src.write(prod_list+'.'+reag_list+'\n')
        valid_b_tgt.write(reac_list+'\n')
    
    for k in tqdm ( range(0, len(test))):
        row = test[k][:test[k].find('|')]
        reag_list=row.strip().split(',')[1].split('>')[0]
        prod_list=row.strip().split(',')[1].split('>')[1]
        
        if space:
            prod_list   = ' '.join(prod_list)
            reag_list   = ' '.join(reag_list)
        
        vocab = set(reag_list) | set(prod_list) | vocab
        
        if space:
            src_len.append(len( prod_list+' . '+reag_list+'\n' ))
        else:
            src_len.append(len( prod_list+'.'+reag_list+'\n' ))
        
        if space:
            test_src.write(prod_list+' . '+reag_list+'\n')
        else:
            test_src.write(prod_list+'.'+reag_list+'\n')
    
    for elem in tqdm( vocab ):
        vocab_file.write(elem+'\n')

    print('max len of src: ',max(src_len), ' mean len of src: ',sum(src_len)/len(src_len))
    print('max len of tgt: ',max(tgt_len), ' mean len of tgt: ',sum(tgt_len)/len(tgt_len))
    
    train_src.close()
    train_tgt.close()
    valid_a_src.close()
    valid_a_tgt.close()
    valid_b_src.close()
    valid_b_tgt.close()
    test_src.close()
    vocab_file.close()

def eval(test_gold, test_pred):
    with open(test_gold, 'r') as fgold:
        gold_lines = fgold.readlines()
    with open(test_pred, 'r') as fpred:
        pred_lines = fpred.readlines()

    assert(len(gold_lines) == len(pred_lines))
    
    test_num    = len(gold_lines)
    exact_num   = 0
    f1_all      = 0.0
    
    exact       = 0.0
    f1          = 0.0

    for i in tqdm( range(test_num) ):
        
        gold_line = gold_lines[i].strip()
        pred_line = pred_lines[i].strip()  

        exact_num   += compute_exact_(gold_line, pred_line)
        f1_all      += compute_f1_(gold_line, pred_line) 
    
    exact       = 1.0 * exact_num / test_num
    f1          = 1.0 * f1_all / test_num
    score       = 0.75 * f1 + 0.25 * exact
    
    return 'f1: '+str(f1), 'em:  '+str(exact), 'score:    '+str(score)

if __name__ =='__main__':
    
    #data_split()

    print(eval('../kescidata/valid_b_targets_2000', '../train/1/valid_b_targets_5000'))
    
    #vis_test()

    smiles0 = 'CCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4'
    smiles1 = 'CCCc1nc(C)c2C(=O)N=C(Nn12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(CC)CC4'
    #vis_rdkit(smiles1)
