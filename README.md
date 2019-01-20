# Retrosynthetic-Reaction-Prediction

2019"默克"杯逆合成反应预测大赛

[Competition URL](https://www.kesci.com/home/competition/merck?from=mpdf)

Model: Transformer

Framework: tensor2tensor

Data format: SMILES

Visualize Tool: RDKit

### Evaluation:

|modelName|framework |score | f1 | em |
| ------ |------|------ |------ | ------ |
|transformer-base| tensor2tensor |0.627 | 0.764 | 0.218 |
|transformer-merge| tensor2tensor |0.636 | 0.770 | 0.235 |
|transformer-base| opennmt-py |0.646 | 0.860 | 0.002 |
|transformer-base-200000| opennmt-py |0.650 | 0.865 | 0.004 |
|transformer-base-254000| opennmt-py |0.653 | 0.869 | 0.005 |
|transformer-base-400000| opennmt-py |0.660 | 0.877 | 0.007 |
