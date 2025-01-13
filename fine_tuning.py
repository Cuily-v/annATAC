import sys
import numpy as np
from numpy import array
import pandas as pd
import os
from linformer_pytorch import LinformerLM
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import warnings
import random
import torch, gc
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
import tracemalloc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import datetime
import argparse
import scanpy as sc
warnings.filterwarnings("ignore")

data_pretrain = sc.read_h5ad("../fintune2.h5ad" )
count_list = data_pretrain.X

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0,1,2')
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument('--batchSize', type=int, default=5)
parser.add_argument('--Learn_rate', type=int, default=1e-4)
parser.add_argument('--d_model', type=int, default=200)
parser.add_argument('--maxlen', type=int, default=16656)                 
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--gamma', type=int, default=0.3)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--grad_acc', type=int, default=30)
parser.add_argument('--valid_every', type=int, default=1)
parser.add_argument('--model_name', type=str, default='fintune')
config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
world_size = 3 
SEQ_LEN = config.maxlen + 1
CLASS = config.bin_num + 2   

def distributed_concat(tensor, num_total_examples, world_size):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

class SCDataset(Dataset):
    def __init__(self, data,label):
        super().__init__()
        self.data = data
        self.label = label
    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label
    def __len__(self):
        return self.data.shape[0]

class out_CNN(nn.Module):
    def __init__(self, dropout = 0, h_dim = 100, out_dim = 10):
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

label_infor = pd.read_csv('../data/fintune_label.csv')
label_dict = pd.unique(label_infor.iloc[:,11])  
char_to_num = {}
num = 1
for char in pd.Series(label_infor.iloc[:,11]):
    if char not in char_to_num:
        char_to_num[char] = num
        num += 1
label = array(pd.Series(label_infor.iloc[:,11]).map(char_to_num))
label = torch.from_numpy(label)-1
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
data = count_list

acc = []
f1 = []
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config.seed)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)
print(f"training data shape：{data_train.shape}")
print(f"Validate data shape：{data_val.shape}")
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=config.batchSize, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=config.batchSize, sampler=val_sampler)

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

path = config.model_path
ckpt = torch.load(path)
new_dict = {key.replace("module.", ""): value for key, value in ckpt.items()}  
model.load_state_dict(new_dict)
for param in model.parameters():            
    param.requires_grad = True
model.to_logits = out_CNN(dropout=0, h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = torch.optim.Adam(model.parameters(), lr=config.Learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=config.gamma)
criterion = nn.CrossEntropyLoss(weight=class_weight).to(local_rank)
softmax = nn.Softmax(dim=-1)

dist.barrier()
trigger_times = 0
max_acc = 0.0
for i in range(1, config.num_epochs):
    print(f'Epoch: {i}')
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    train_loss = 0.0
    cum_acc = 0.0
    f1_sc = 0.0
    for index, (batch_data, train_label) in enumerate(train_loader): 
        if batch_data.shape[0] < config.batchSize:
            break
        index += 1
        if index % 5000 == 0:
            print(f'========= Train Epoch: {i} |  loader: {index}  ========')
        batch_data, train_label = batch_data.to(device), train_label.to(device)
        if index % config.grad_acc != 0:
            with model.no_sync(): 
                logits = model(batch_data)
                loss = criterion(logits, train_label)
                loss.backward()
        if index % config.grad_acc == 0:
            logits = model(batch_data)
            loss = criterion(logits, train_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item()
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = train_label.size(0)
        correct_num = torch.eq(final, train_label).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        final = final.cpu().numpy().flatten()
        train_label = train_label.cpu().numpy().flatten()
        f1 = f1_score(train_label, final, average='macro')
        f1_sc += f1
    train_loss = train_loss / index
    train_acc = 100 * cum_acc / index
    train_f1 = 100 * f1_sc / index
    train_loss = get_reduced(train_loss, local_rank, 0, world_size)   
    train_acc = get_reduced(train_acc, local_rank, 0, world_size)
    train_f1 = get_reduced(train_f1, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {train_loss:.6f}  ==')
        print(f'    ==  Epoch: {i} | Training ACC: {train_acc:.6f} %  ==')
        print(f'    ==  Epoch: {i} | Training F1_score: {train_f1:.6f} %  ==')
    dist.barrier()
    scheduler.step()
    del batch_data,train_label,logits,final

    if i % config.valid_every == 0:
        model.eval()
        dist.barrier()
        val_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for inds, (val_batch_data, val_label) in enumerate(val_loader):
                if val_batch_data.shape[0] < config.batchSize:
                    break
                inds += 1
                if inds % 5000 == 0:
                    print(f'========= Val Epoch: {i} |  loader: {inds}  ========')
                val_batch_data, val_label = val_batch_data.to(device), val_label.to(device)
                logits = model(val_batch_data)
                loss = criterion(logits, val_label)
                val_loss += loss.item()
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < 0] = -1
                predictions.append(final)
                truths.append(val_label)
            del val_batch_data, val_label,logits, final_prob, final
            torch.cuda.empty_cache()
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = 100*accuracy_score(truths, predictions)
            f1 = 100* f1_score(truths, predictions, average='macro')
            val_loss = val_loss / inds
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.2f}   ==')
                print(f'    ==  Epoch: {i} | f1_score: {f1:.2f}   ==')
                print(f'    ==  Epoch: {i} | cur_acc: {cur_acc:.2f}   ==')
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times > 10:
                    break
            del predictions, truths
