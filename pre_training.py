import sys
import numpy as np
from numpy import array
import pandas as pd
import math
import os
import torch.utils.data as Data
import torch.nn as nn
from functools import reduce
from sklearn.model_selection import train_test_split
from linformer_pytorch import LinformerLM
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import warnings
import random
import time
import torch, gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
import tracemalloc
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import scanpy as sc
import datetime
import psutil
import argparse
from sklearn.metrics import f1_score

# 1 # load data 
data_pretrain = sc.read_h5ad("./data/train_data.h5ad")
count_list = data_pretrain.X

# 2 # setting parameter 
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5')
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument('--batchSize', type=int, default=5)
parser.add_argument('--Learn_rate', type=int, default=1e-4)
parser.add_argument("--mask_prob", type=float, default=0.15)
parser.add_argument("--replace_prob", type=float, default=0.9)
parser.add_argument('--d_model', type=int, default=200)
parser.add_argument('--maxlen', type=int, default=16656)                 
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--gamma', type=int, default=0.9)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--grad_acc', type=int, default=30)
parser.add_argument('--valid_every', type=int, default=1)
parser.add_argument('--model_name', type=str, default='scATAC-seq_ann')
config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
world_size = 6
SEQ_LEN = config.maxlen + 1
CLASS = config.bin_num + 2  
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]


# 3 # mask data #
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob,data):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_num = (num_tokens * prob).ceil()
    new_mask = torch.zeros((batch, seq_len), device=device)
    for row in range(batch):
        if mask_num[row] > max_masked:
            mask_num[row] = max_masked
        num_class = len(torch.unique(data[row],return_counts=True)[1])-1
        min_mask_num_per_class = torch.floor(mask_num[row] / num_class)
        classes = torch.unique(data[row],return_counts=True)[0]
        classes_indices =[]
        for cls in classes:
            if cls == 0:
                continue
            else:
                classes_indices.append(torch.where(data[row] == cls))
        for class_indices in classes_indices:
            if min_mask_num_per_class > len(class_indices[0]):
                min_mask_num_per_class = len(class_indices[0])
            if min_mask_num_per_class == 0:
                min_mask_num_per_class = 1
            random_indices = torch.randperm(len(class_indices[0]))[:int(min_mask_num_per_class)]
            class_masked_indices = class_indices[0][random_indices]
            new_mask[row][class_masked_indices] = 1
    return new_mask.bool()

def data_mask(data,
    mask_prob = 0.15,
    replace_prob = 0.9,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)
    mask = get_mask_subset_with_prob(~no_mask, mask_prob,data)
    masked_input = data.clone().detach()
    replace_prob = prob_mask_like(data, replace_prob)
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)   # mask * 0.9
    labels = data.masked_fill(~mask, pad_token_id)
    return masked_input, labels

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

data_train, data_val = train_test_split(count_list, test_size=0.01, random_state=64)
print(f"training data shape：{data_train.shape}")
print(f"Validate data shape：{data_val.shape}")
train_dataset = SCDataset(data_train)
val_dataset = SCDataset(data_val)
train_sampler = DistributedSampler(train_dataset)
val_sampler = SequentialDistributedSampler(val_dataset, batch_size=config.batchSize, world_size=world_size)
train_loader = DataLoader(train_dataset, batch_size=config.batchSize, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=config.batchSize, sampler=val_sampler)

# 4 # model #
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

model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = torch.optim.Adam(model.parameters(), lr=config.Learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=config.gamma)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean').to(device)
softmax = nn.Softmax(dim=-1)

# 5 # train 
print("===begin train===")
dist.barrier()                                               
for i in range(1, config.num_epochs):
    print(f'Epoch: {i}')
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    train_loss = 0.0
    cum_acc = 0.0
    all_predicted = []
    all_true_labels = []
    for index, batch_data in enumerate(train_loader):            
        if batch_data.shape[0] < config.batchSize:
            break
        index += 1
        if index % 10000 == 0:
            print(f'========= Train Epoch: {i} |  loader: {index}  ========')
        batch_data = batch_data.to(device)
        data_tra, labels_tra = data_mask(batch_data)
        if index % config.grad_acc != 0:
            with model.no_sync():
                logits_tra = model(data_tra)
                loss_tra = criterion(logits_tra.transpose(1, 2), labels_tra)/config.grad_acc
                loss_tra.backward()
        if index % config.grad_acc == 0:
            logits_tra = model(data_tra)
            loss_tra = criterion(logits_tra.transpose(1, 2), labels_tra) / config.grad_acc
            loss_tra.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss_tra.item()
        final_tra = softmax(logits_tra)[..., 1:-1]
        final_tra = final_tra.argmax(dim=-1) + 1
        pred_num = (labels_tra != PAD_TOKEN_ID).sum(dim=-1)
        correct_num = ((labels_tra != PAD_TOKEN_ID) * (final_tra == labels_tra)).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        mask = labels_tra != PAD_TOKEN_ID
        true_labels_filtered = labels_tra[mask].cpu().numpy()
        predicted_filtered = final_tra[mask].cpu().numpy()
        all_predicted.extend(predicted_filtered)
        all_true_labels.extend(true_labels_filtered)
    del data_tra, labels_tra, logits_tra, final_tra
    cum_acc = 100 * cum_acc / index
    train_loss = train_loss / index
    train_loss = get_reduced(train_loss, local_rank, 0, world_size)
    cum_acc = get_reduced(cum_acc, local_rank, 0, world_size)
    f1 = f1_score(all_true_labels, all_predicted, average='macro')
    print(f"F1 Score for this epoch: {f1}")
    if is_master:
        print(f'    ==  Epoch: {i} | Train Loss: {train_loss:.6f} | Accuracy: {cum_acc:6f}%  ==')
    dist.barrier()
    scheduler.step()

    if i % config.valid_every == 0:
        model.eval()
        dist.barrier()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for inds, val_batch_data in enumerate(val_loader):
                if val_batch_data.shape[0] < config.batchSize:
                    break
                inds += 1
                if inds % 10000 == 0:
                    print(f'========= Val Epoch: {i} |  loader: {inds}  ========')
                val_batch_data = val_batch_data.to(device)
                data, labels = data_mask(val_batch_data)
                logits = model(data)
                loss = criterion(logits.transpose(1, 2), labels)
                val_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final = softmax(logits)[..., 1:-1]
                final = final.argmax(dim=-1) + 1
                val_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
                correct_num_val = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
                val_acc += torch.true_divide(correct_num_val, val_num).mean().item()
            del data, labels, logits, final
            val_acc = 100 * val_acc / inds
            val_loss = val_loss / inds
            val_acc = get_reduced(val_acc, local_rank, 0, world_size)
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')














