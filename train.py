import os
import sys
import torch
import pandas as pd
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import argparse
# from transformers import BertTokenizer

from model.hremodel import HRE
from utils.metrics import acc, cen, mcc
from utils.losses import xe, cl_pos, cl_neg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--train_path', type=str, default='./data/schneider50k_train.tsv', help='path to the training data')
    parser.add_argument('--test_path', type=str, default='./data/schneider50k_test.tsv', help='path to the test data')
    parser.add_argument('--vocab_file', type=str, default='./data/vocab', help='path to the vocab')
    
    
    # model setting
    parser.add_argument('--truncation_length', type=int, default=300, help='max input length')
    parser.add_argument('--d_model', type=int, default=256, help='hidden dimension')
    parser.add_argument('--position_embedding', type=int, default=0)
    parser.add_argument('--class_num', type=int, default=50)
    parser.add_argument('--superclass_num', type=int, default=9)
    parser.add_argument('--fineclass_num', type=int, default=50)
    
    # optimization setting
    parser.add_argument('--lr', type=float, default=0.3, help='learning rate')
    parser.add_argument('--step_size', type=float, default=15, help='optimization step size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    
    
    args = parser.parse_args()



class SentimentDataset(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep="\t", index_col=0, names=["text", "label", "superclass", "fineclass"])
        self.dataset.drop(self.dataset.index[0], inplace=True)
    def __len__(self):
        return len(self.node)
    def __getitem__(self, idx):
        text = self.dataset.iloc[idx, 0]
        label = self.dataset.iloc[idx, 1]
        label = pd.to_numeric(label)
        superclass = self.dataset.iloc[idx, 2]
        superclass = pd.to_numeric(superclass)
        fineclass = self.dataset.iloc[idx, 3]
        fineclass = pd.to_numeric(fineclass)
        sample = {"text": text, "label": label, "superclass": superclass, "fineclass": fineclass}
        return sample

def train_one_epoch(train_dataloader, model, optimizer, epoch, tokenizer, args):
    model.train()
    loss_list = []
    # device = model.parameters()).device
    for idx, batch in enumerate(train_dataloader):
        label = batch['label'].to(device)
        superclass = batch['superclass'].to(device)
        fineclass = batch['fineclass'].to(device)
        label = label.view(-1).to(torch.float32)
        superclass = superclass.view(-1).to(torch.float32)
        fineclass = fineclass.view(-1).to(torch.float32)
        
        text = batch['text'].to(torch.float32).to(device)
        # tokenized_text = tokenizer(text, max_length=args.truncation_length, add_special_tokens=True, truncation=True, padding='max_length', return_tensors="pt")
        # tokenized_text = tokenized_text.to(device)
        cls_org, z_org, p_org, alpha_org =  model(text)
        cls_aug, z_aug, p_aug, _ =  model(text, superclass = superclass, alpha = alpha_org, tau = 3*(0.5-epoch/args.epochs))
        
        loss_pos = 0.5*(cl_pos(p_org[0], z_aug[0])+cl_pos(p_aug[0],z_org[0]))/2+0.5(cl_pos(p_org[1], z_aug[1])+cl_pos(p_aug[1],z_org[1]))/2
        loss_neg = cl_neg(z_org[1],z_aug[1])
        loss_cls = xe(label,cls_org[0])+0.5*xe(superclass,cls_org[1])+0.5*xe(fineclass,cls_org[2])+xe(label,cls_org[3])
        loss_cls += (xe(label,cls_aug[0])+xe(label,cls_aug[3])+0.5*xe(cls_aug,cls_org[1])+0.5*xe(cls_aug,cls_org[2]))/2
        loss = 0.1*loss_pos+0.01*loss_neg+loss_cls
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % args.print_freq == 0:
            print('epoch: {}, batch: {}, loss: {:.5f}'.format(epoch, idx, np.mean(loss)))
            
def test_epoch(test_dataloader, model, epoch, tokenizer, args):
    model.eval()

    global_loss_list = []
    coarse_loss_list = []
    coarse_prediction_list = []
    prediction_list = []
    label_list = []
    superclass_list = []

    # device = model.parameters()).device
    for idx, batch in enumerate(test_dataloader):
        label = batch['label'].to(device)
        superclass = batch['superclass'].to(device)
        # text = batch['text']
        text = batch['text'].to(torch.float32).to(device)
        label = label.view(-1).to(torch.float32)
        superclass = superclass.view(-1).to(torch.float32)
        # tokenized_text = tokenizer(text, max_length=Truncation_length, add_special_tokens=True, truncation=True, padding='max_length', return_tensors="pt")
        # tokenized_text = tokenized_text.to(device)
        
        with torch.no_grad():
            cls, _, _, _ =  model(text)
        loss_global = xe(label,cls[0])+xe(label,cls[3])
        loss_coarse = xe(superclass,cls[1])
        global_loss_list.append(loss_global.item())
        coarse_loss_list.append(loss_coarse.item())
        
        coarse_prediction_list.append(cls[1].argmax(dim=1))
        prediction_list.append((cls[0]+cls[3]).argmax(dim=1))
        label_list.append(label)
        superclass_list.append(superclass)
    
    loss_global = np.mean(global_loss_list)
    loss_coarse = np.mean(coarse_loss_list)
    print('cls_loss: {:.5f}, coarse_loss: {:.5f}'.format(loss_global, loss_coarse))
    
    coarse_prediction_list = torch.cat(coarse_prediction_list, dim = 0)
    prediction_list = torch.cat(prediction_list, dim = 0)
    label_list = torch.cat(label_list, dim = 0)
    superclass_list = torch.cat(superclass_list, dim = 0)
    
    acc_global = acc(label_list,prediction_list)
    acc_coarse = acc(superclass_list,coarse_prediction_list)
    print('cls_acc: {:.5f}, coarse_acc: {:.5f}'.format(acc_global, acc_coarse))
    cen_global = cen(label_list,prediction_list)
    cen_coarse = cen(superclass_list,coarse_prediction_list)
    print('cls_cen: {:.5f}, coarse_cen: {:.5f}'.format(cen_global, cen_coarse))
    mcc_global = mcc(label_list,prediction_list)
    mcc_coarse = mcc(superclass_list,coarse_prediction_list)
    print('cls_mcc: {:.5f}, coarse_mcc: {:.5f}'.format(mcc_global, mcc_coarse))
    
        

def launch_training():
    args = parse_option()
    print("{}".format(args).replace(', ', ',\n'))
    
    
    
    train_data = SentimentDataset(args.train_path)
    test_data = SentimentDataset(args.test_path)

    train_dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
    test_dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
    # tokenizer = BertTokenizer.from_pretrained(vocab_file)
    
    model = HRE(args = args)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_steplr = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_steplr)
    
    for epoch in range(args.epochs):
        tb = time.time()
        # train_one_epoch(train_dataloader, model, optimizer, epoch, tokenizer, args)
        train_one_epoch(train_dataloader, model, optimizer, epoch, args)
        te = time.time()
        print('training epoch {} cost {:.2f} min'.format(epoch, (te - tb)/60)
        
        test_epoch(test_dataloader, model, epoch, args)
        # test_epoch(test_dataloader, model, epoch, tokenizer, args)
    
    
    
    

if __name__ == '__main__':
    launch_training()
