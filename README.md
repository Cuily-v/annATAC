# annATAC
Here, we introduce a novel method named annATAC, which is designed for the automatic annotation of cell types in scATAC-seq data based on pre-training. This method primarily consists of three key components. During the pre-training phase, by training on a vast amount of unlabeled data, the model can learn the interaction relationships between peaks, thus building a preliminary understanding of the data features. Subsequently, in the fine-tuning stage, a small quantity of labeled data is utilized to conduct secondary training on the model. This enables the model to acquire the capability to accurately identify cell types. Finally, in the prediction stage, the trained model is applied to annotate scATAC-seq data. When compared with other automatic annotation methods across multiple datasets, annATAC demonstrates remarkable superiority in the annotation task.

![GRAPHICAL_ABSTRACT](https://github.com/user-attachments/assets/bb0a50c1-870a-4b39-b264-f56b4463b420)
## Data
The pre-training data, demo data (labels) used in this article can be downloaded and obtained from the following link. https://drive.google.com/drive/folders/1ikvAgWNA1MyIrVU64Z7h_C2_O-LvMNCG.
## Demo
In the demo folder, we provide an example for predicting the subtypes of Alzheimer's disease cells. The data folder contains the cell labels provided in the original paper. After running the AD_predict.py file, the accuracy of the model's prediction can be determined by comparing it with the original labels.
## model 
The pre-training and fine-tuning models after training can be obtained from the following link. https://drive.google.com/drive/folders/1UPtVtK3WOdlxEGzkfuNnDQuISFqcXm0z.
