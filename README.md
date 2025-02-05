# annATAC
Here, we introduce a novel method named annATAC, which is designed for the automatic annotation of cell types in scATAC-seq data based on language model. This method primarily consists of three key components. Firstly, pre-training. The peak-cell matrix of a large amount of unlabeled scATAC-seq data is used for pre-training, enabling the model to fully learn the interaction relationships between peaks. Secondly, fine-tuning. A small amount of labeled data is employed to conduct secondary training on the pre-trained model, so that the model is fully formed. Finally, predict cell types. Cell-type prediction is performed on the unlabeled scATAC-seq data. 

![GRAPHICAL_ABSTRACT](https://github.com/user-attachments/assets/b81a0602-44fc-4be1-b94a-fc842e163316)

## Data
The pre-training data, demo data (label) used in this study can be downloaded and obtained from the following link. https://drive.google.com/drive/folders/1ikvAgWNA1MyIrVU64Z7h_C2_O-LvMNCG.
## Demo
In the demo folder, we provide an example for predicting the subtypes of Alzheimer's disease cells. The data folder contains the cell labels provided in the original paper. After running the AD_predict.py file, the accuracy of the model's prediction can be determined by comparing it with the original labels.
## Model 
The pre-training and fine-tuning models after training can be obtained from the following link. https://drive.google.com/drive/folders/1UPtVtK3WOdlxEGzkfuNnDQuISFqcXm0z.
