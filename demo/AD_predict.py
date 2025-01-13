import sys
from numpy import array
import pandas as pd
import os
from linformer_pytorch import Linformer, LinformerLM
import random
import torch, gc
import torch.nn as nn
import torch.distributed as dist
from utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import argparse
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.metrics import jaccard_score

data = pd.read_csv("../data/AD.csv")
data = data.iloc[:, 1:]
label_infor = pd.read_csv('../data/fintune_label.csv')
label_dict = pd.unique(label_infor.iloc[:,11])
label_infor_data = pd.read_csv("../data/label.csv")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0,1,2')
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--d_model', type=int, default=200)
parser.add_argument('--maxlen', type=int, default=16656)
parser.add_argument("--novel_type", type=bool, default=False) 
parser.add_argument("--unassign_thres", type=float, default=0.5)
parser.add_argument('--model_path', type=str, default='../model/fine_tuning_para.pth')
config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
world_size = 3
UNASSIGN = config.novel_type
UNASSIGN_THRES = config.unassign_thres if UNASSIGN == True else 0
SEQ_LEN = config.maxlen + 1
CLASS = config.bin_num + 2

class out_CNN(nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(out_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = LinformerLM(
    num_tokens=CLASS,
    input_size=SEQ_LEN,
    channels=config.d_model, 
    dim_ff=(config.d_model)*4,
    dim_d=128, 
    nhead=4, 
    depth=4, 
    activation="gelu",
    peak_svd_embedding = True
)
model.to_logits = out_CNN(dropout=0, h_dim=128, out_dim=label_dict.shape[0])
path = config.model_path
ckpt = torch.load(path)
new_dict = {key.replace("module.", ""): value for key, value in ckpt.items()}
model.load_state_dict(new_dict)
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)
batch_size = data.shape[0]
model.eval()
pred_finals = []
novel_indices = []
with torch.no_grad():
    for index in range(batch_size):
        full_seq = array(data.iloc[index,:])
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_final = pred_prob.argmax(dim=-1).item()
        if np.amax(np.array(pred_prob.cpu()), axis=-1) < UNASSIGN_THRES:
            novel_indices.append(index)
        pred_finals.append(pred_final)
pred_list = label_dict[pred_finals].tolist()
for index in novel_indices:
    pred_list[index] = 'New_cell_type'
label = pd.DataFrame(label_infor_data.iloc[:,11])
print(data.shape)
print(label.shape)
cur_acc = 100*accuracy_score(label, pred_list)
print("ACC:",cur_acc)
f1 = 100* f1_score(label, pred_list,average='macro')
print("F1:",f1)
kappa_value = cohen_kappa_score(label, pred_list)
print("Cohen's Kappa:",kappa_value)
label_array = label.values.flatten()
pred_array = np.array(pred_list)
jaccard_weighted = jaccard_score(label_array, pred_array, average='weighted')
jaccard_micro = jaccard_score(label_array, pred_array, average='micro')
print("jaccard_weighted:",jaccard_weighted)
print("jaccard_micro:",jaccard_micro)



