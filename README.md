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
|transformer-base| opennmt-py |0.646 | 0.860 | 0.002 |
