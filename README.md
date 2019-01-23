# Retrosynthetic-Reaction-Prediction

2019"默克"杯逆合成反应预测大赛

**The work is ongoing!!!**

[Competition URL](https://www.kesci.com/home/competition/merck?from=mpdf)

Model: Transformer/RNN

Framework: tensor2tensor/opennmt-py

Data format: SMILES

Visualize Tool: RDKit


### Evaluation:

Evaluation Method is the same as SQuAD dataset, [code here](https://worksheets.codalab.org/rest/bundles/0x4c6febb3f9574587a6729b23b5e2f290/contents/blob/)


#### Data from [《Retrosynthetic Reaction Prediction Using Neural Sequence to Sequence Models》](https://github.com/pandegroup/reaction_prediction_seq2seq)


|modelName|framework |score | f1 | em |
| ------ |------|------ |------ | ------ |
|transformer-base| tensor2tensor |0.627 | 0.764 | 0.218 |
|transformer-merge| tensor2tensor |0.636 | 0.770 | 0.235 |
|transformer-base| opennmt-py |0.646 | 0.860 | 0.002 |
|transformer-base-200000| opennmt-py |0.650 | 0.865 | 0.004 |
|transformer-base-254000| opennmt-py |0.653 | 0.869 | 0.005 |
|transformer-base-400000| opennmt-py |0.660 | 0.877 | 0.007 |
|transformer-base-800000| opennmt-py |0.664 | 0.881 | 0.013 |
|rnn-based-115000| opennmt-py |0.415 | 0.554 | 0.000 |

#### Data from "默克杯"比赛:

|modelName|framework |score | f1 | em |
| ------ |------|------ |------ | ------ |
|rnn-based-6000| opennmt-py |0.500 | 0.663 | 0.000 |

#### Useful Paper and Code:

《Retrosynthetic Reaction Prediction Using Neural Sequence-to-Sequence Models》

《Found in Translation Predicting Outcomes of Complex Organic Chemistry Reactions using Neural Sequence to Sequence Models》


