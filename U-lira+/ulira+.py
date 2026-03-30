from transformers import  AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import numpy as np

import pandas as pd
from sklearn.metrics import  auc
import pickle
import os 

import warnings
from sklearn.metrics import accuracy_score

import torch.nn.functional as F

from dataset import TextDataset_cls
from functions import InOutTest, sweep
warnings.filterwarnings("ignore")

from copy import deepcopy
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(13)

def unlearn_cls(model,tokenizer,unlearn_set,retain_set,unlearn_md="ga",lr=None,ep=None,bs=32):
    device=model.device
    ul_model=deepcopy(model)
    ul_model.eval()
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 512
     
    


    unlearn_num=unlearn_set.getlen()
    num_batch=unlearn_num//bs
    if num_batch==0:
        bs= unlearn_num

    optimizer = torch.optim.AdamW(ul_model.parameters(), lr=lr)

    for e in range(ep):
        ul_model.train()
        step = 0
        train_loss = 0.
        total_sample = 0
        all_train_preds = []
        all_train_labels = []
        print(unlearn_md)
        for i in range(0,unlearn_num//bs):
            #print(unlearn_md)
            step += 1
            if i*bs+bs>unlearn_num-1:
                seqs,labels=unlearn_set[i*bs:unlearn_num]
            else:
                seqs,labels=unlearn_set[i*bs:i*bs+bs]

            labels=torch.tensor(labels).to(device)

            seqs=[seq[0] for seq in seqs]

            orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=50,return_tensors='pt').to(device)

            optimizer.zero_grad()
            outs = ul_model(input_ids=orig_batch['input_ids'], labels=labels,attention_mask=orig_batch['attention_mask'])
            if unlearn_md=="ga":
                print(unlearn_md)
                gd=torch.autograd.grad((-1)*outs.loss, [param for param in ul_model.parameters() if param.requires_grad], allow_unused=True)
            elif unlearn_md=="kl":
                print(unlearn_md)
                with torch.no_grad():
                    o_outs = model(input_ids=orig_batch['input_ids'], labels=labels,attention_mask=orig_batch['attention_mask'])
                gd=torch.autograd.grad((-1)*F.kl_div(outs.logits,o_outs.logits,reduction = 'batchmean',log_target = True), [param for param in ul_model.parameters() if param.requires_grad], allow_unused=True)
            elif unlearn_md=="npo":
                print(unlearn_md)
                beta=0.1
                with torch.no_grad():
                    o_outs = model(input_ids=orig_batch['input_ids'], labels=labels,attention_mask=orig_batch['attention_mask'])
                neg_log_ratio = o_outs.logits - outs.logits
                gd=torch.autograd.grad(-F.logsigmoid(beta * neg_log_ratio).mean() * 2 / beta, [param for param in ul_model.parameters() if param.requires_grad], allow_unused=True)
            elif unlearn_md=="taskVec":
                print(unlearn_md)
                gd=torch.autograd.grad(outs.loss, [param for param in ul_model.parameters() if param.requires_grad], allow_unused=True)
            #print(gd)
            for param, grad in zip(ul_model.parameters(), gd):
                if param.requires_grad and grad is not None:
                    #print("yes")
                    #exit()
                    param.data -=lr * grad
            optimizer.step()
            train_loss += bs * outs.loss.item()
            total_sample += bs

            preds = torch.argmax(outs.logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            if step % 1== 0:
                cur_t_loss = train_loss / total_sample
                train_accuracy = accuracy_score(all_train_labels, all_train_preds)
                print('Unlearn_Epoch {:5d} | Train loss {:4.4f} | Train Accuracy {:4.4f} | {:5d}/{:5d} batches'.format(e, cur_t_loss, train_accuracy, step, num_batch))
                train_loss = 0.
                total_sample = 0
                all_train_preds = []
                all_train_labels = []
   

        if e>=0:
            ul_model.eval()
            unlearn_num = unlearn_set.getlen()
            unlearn_num_batch = unlearn_num // bs
            all_preds = []
            all_labels = []
            if unlearn_num_batch==0:
                eval_bs=unlearn_num
            else:
                eval_bs=bs
            with torch.no_grad():
                for i in range(0, unlearn_num // eval_bs):
                    if i * eval_bs + eval_bs > unlearn_num- 1:
                        seqs, labels = unlearn_set[i * eval_bs:unlearn_num]
                    else:
                        seqs, labels = unlearn_set[i * eval_bs:i * eval_bs + eval_bs]
                    
                    labels = torch.tensor(labels).to(device)
                    seqs = [seq[0] for seq in seqs]
                    orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=50, return_tensors='pt').to(device)
                    
                    outputs = ul_model(input_ids=orig_batch['input_ids'], attention_mask=orig_batch['attention_mask'])
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            #print(all_labels, all_preds)
            accuracy = accuracy_score(all_labels, all_preds)
            print('Unlearn Accuracy: {:.4f}'.format(accuracy))
            u_acc=accuracy
            retain_num = retain_set.getlen()
            retain_num_batch = retain_num // bs
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for i in range(0, retain_num // bs):
                    if i * bs + bs > retain_num - 1:
                        seqs, labels = retain_set[i * bs:retain_num]
                    else:
                        seqs, labels = retain_set[i * bs:i * bs + bs]
                    
                    labels = torch.tensor(labels).to(device)
                    seqs = [seq[0] for seq in seqs]
                    orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=50, return_tensors='pt').to(device)
                    
                    outputs = ul_model(input_ids=orig_batch['input_ids'], attention_mask=orig_batch['attention_mask'])
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            print('Retain Accuracy: {:.4f}'.format(accuracy))

    if unlearn_md=="taskVec":
        for param, o_param in zip(ul_model.parameters(), model.parameters()):
            param.data =o_param.data-(param.data-o_param.data)
        ul_model.eval()
        unlearn_num = unlearn_set.getlen()
        unlearn_num_batch = unlearn_num // bs
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, unlearn_num // bs):
                if i * bs + bs > unlearn_num- 1:
                    seqs, labels = unlearn_set[i * bs:unlearn_num]
                else:
                    seqs, labels = unlearn_set[i * bs:i * bs + bs]
                
                labels = torch.tensor(labels).to(device)
                seqs = [seq[0] for seq in seqs]
                orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=50, return_tensors='pt').to(device)
                
                outputs = ul_model(input_ids=orig_batch['input_ids'], attention_mask=orig_batch['attention_mask'])
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print('Unlearn Accuracy: {:.4f}'.format(accuracy))

        retain_num = retain_set.getlen()

        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, retain_num // bs):
                if i * bs + bs > retain_num - 1:
                    seqs, labels = retain_set[i * bs:retain_num]
                else:
                    seqs, labels = retain_set[i * bs:i * bs + bs]
                
                labels = torch.tensor(labels).to(device)
                seqs = [seq[0] for seq in seqs]
                orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=50, return_tensors='pt').to(device)
                
                outputs = ul_model(input_ids=orig_batch['input_ids'], attention_mask=orig_batch['attention_mask'])
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print('Retain Accuracy: {:.4f}'.format(accuracy))
    del model
    return ul_model


args = argparse.ArgumentParser() 
args.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)") 
args.add_argument("--dataset", type=str, default="dataset_path", help="Path to dataset") 
args.add_argument("--batch_size", type=int, default=64, help="Batch size") 
args.add_argument("--dataset_size", type=int, default=1000, help="Data size") #age3200 inc2400 
args.add_argument("--test_dataset_size", type=int, default=200, help="Data size") #age400 inc100
args.add_argument('--model_name', type=str, default='bert-base-uncased')
args.add_argument('--train_lr', type=float, default=1e-5) 
args.add_argument('--train_ep', type=int, default=2)
args.add_argument('--mis_label', type=int, default=0)#mis_label 0/1,1 is our u-lira+
args.add_argument('--unlearn_ep', type=int, default=2)
args.add_argument('--unlearn_md', type=str, default="ga")
args.add_argument('--num_unlearn', type=int, default=0)#32,24

args = args.parse_args()
device=args.device

dataset_path = f"{args.dataset}_{args.dataset_size}_train.pkl"
test_dataset_path = f"{args.dataset}_{args.test_dataset_size}_test.pkl"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.model_max_length = 512
# Check if the dataset has been saved
if os.path.exists(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
else:
    
    dataset = TextDataset_cls(args.device, args.dataset, dataset_size=args.dataset_size, batch_size=1, split="train")

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

test_dataset = TextDataset_cls(args.device, args.dataset, dataset_size=args.test_dataset_size, batch_size=1, split="test")#100

with open(test_dataset_path, "wb") as f:
    pickle.dump(test_dataset, f)

'''
keep = np.random.uniform(0,1,size=(100, args.num_unlearn*2))
order = keep.argsort(0)
keep = order < int(50)
keep = np.array(keep, dtype=bool)
'''
#np.save(f'keeps/keep_{args.model_name}_{args.dataset}.npy', keep)
#exit()

keep = np.load(f'keeps/keep_{args.model_name}_{args.dataset}.npy')

#lira
if args.mis_label==0:
    lira_save_dir=f"preds/pred_ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}"
else:
    lira_save_dir=f"preds/pred_mis_ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}"
''''''
lgts=[]
gt_labels=[]


for id in range(100):#range(100):
    print("Round:", id)
    #lira
    lgts_i=[]
    gt_labels_i=[]
    
    unlearn_idxs=np.where(keep[id] == True)[0].tolist()
    unseen_idxs=np.where(keep[id] == False)[0].tolist()

    retain_idxs=list(set(range(dataset.size))-set(unlearn_idxs+unseen_idxs))

    unlearn_set=dataset.get_subset(unlearn_idxs)

    audit_set=dataset.get_subset(set(unlearn_idxs+unseen_idxs))
    retain_set=dataset.get_subset(retain_idxs)
    if args.mis_label==1:
        unlearn_set.mis_label(list(range(unlearn_set.size)))
        audit_set.mis_label(list(range(audit_set.size)))


    if args.mis_label==0:
        model = AutoModelForSequenceClassification.from_pretrained(f'ft_{args.model_name}_{args.dataset}_{id}.pth',num_labels=2).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f'mis_ft_{args.model_name}_{args.dataset}_{id}.pth',num_labels=2).to(device)
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id
    ul_model=unlearn_cls(model,tokenizer,unlearn_set,retain_set,args.unlearn_md,lr=args.train_lr,ep=args.unlearn_ep,bs=args.batch_size)

    for i in range(audit_set.size):  
        seq,label=audit_set[i]

        orig_batch = tokenizer(seq, padding="max_length", truncation=True, max_length=50,return_tensors='pt').to(device)

        label=label.cpu().to(device)

        with torch.no_grad():

            outs = ul_model(input_ids=orig_batch['input_ids'], labels=label,attention_mask=orig_batch['attention_mask'])

        preds = torch.argmax(outs.logits, dim=1)

        lgts_i.append(outs.logits.cpu().numpy())
        gt_labels_i.append(label.cpu().numpy())
    lgts.append(lgts_i)
    gt_labels.append(gt_labels_i)
    
    
    del unlearn_set,audit_set,retain_set



    if args.mis_label==0:
        ul_model.save_pretrained(f'ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}_{id}.pth')
    else:
        ul_model.save_pretrained(f'mis_ul_{args.model_name}_{args.dataset}_{args.unlearn_md}_{args.unlearn_ep}_{id}.pth')


lgts=np.squeeze(np.array(lgts))#(100,80,feas)
np.save(f'{lira_save_dir}_lgts.npy',lgts)

lgts = np.load(f'{lira_save_dir}_lgts.npy')

predictions = lgts - np.max(lgts, axis=2, keepdims=True)
predictions = np.array(np.exp(predictions), dtype=np.float64)
predictions = predictions/np.sum(predictions,axis=2,keepdims=True)

np.save(f'{lira_save_dir}_predictions.npy', predictions)
predictions = np.load(f'{lira_save_dir}_predictions.npy')
np.save(f'{lira_save_dir}_labels.npy', gt_labels)
gt_labels = np.load(f'{lira_save_dir}_labels.npy')

gt_labels=np.squeeze(gt_labels)


y_true=np.zeros((predictions.shape[0], predictions.shape[1]))

for id in range(100):
    for i in range(predictions.shape[1]):
        y_true[id, i] = predictions[id, i, gt_labels[id,i]]
        #print(gt_labels[0][i])
    for i in range(predictions.shape[1]):
        predictions[id, i, gt_labels[id][i]]=0
y_wrong = np.sum(predictions, axis=2)
score = (np.log(y_true+1e-45) - np.log(y_wrong+1e-45))
ntest=15


prediction, answers = InOutTest(keep[:-ntest],
                             score[:-ntest],
                             keep[-ntest:],
                             score[-ntest:])

fpr, tpr, auc, acc = sweep(np.array(prediction), np.array(answers, dtype=bool))
low = tpr[np.where(fpr<.001)[0][-1]]
print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%("", auc,acc, low))