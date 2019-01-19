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

if __name__ =='__main__':
    
    vis_test()

    smiles0 = 'CCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4'
    smiles1 = 'CCCc1nc(C)c2C(=O)N=C(Nn12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(CC)CC4'
    #vis_rdkit(smiles1)
