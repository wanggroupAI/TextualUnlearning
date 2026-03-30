import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.utils.data as torch_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW,get_scheduler
from datasets import load_dataset
import torch.optim as optim
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import accuracy_score
import logging

BERT_CLS_TOKEN = 101
BERT_SEP_TOKEN = 102
BERT_PAD_TOKEN = 0


def compute_grads(model, x_embeds, y_labels,lr=2e-5,ep=2,unlearn_md="ga", create_graph=False):
    
    ul_model=deepcopy(model)
    for e in range(ep):

        outs = ul_model(inputs_embeds=x_embeds, labels=y_labels)
        

        if unlearn_md=="ga":
            gd=torch.autograd.grad((-1)*outs.loss, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        elif unlearn_md=="kl":
            with torch.no_grad():
                o_outs = model(inputs_embeds=x_embeds.clone().detach(), labels=y_labels)

            gd=torch.autograd.grad((-1)*F.kl_div(outs.logits,o_outs.logits,reduction = 'batchmean',log_target = True), [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)

        elif unlearn_md=="npo":
            beta=0.1
            with torch.no_grad():
                o_outs = model(inputs_embeds=x_embeds.clone().detach(), labels=y_labels)
            neg_log_ratio = o_outs.logits - outs.logits
            gd=torch.autograd.grad(-F.logsigmoid(beta * neg_log_ratio).mean() * 2 / beta, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        elif unlearn_md=="taskVec":

            gd=torch.autograd.grad(outs.loss, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        
    if unlearn_md=="taskVec":
        for g in gd:
            if g is not None:
                g=-g


    return gd 




def get_reconstruction_loss(model, x_embeds, y_labels, true_grads, args, create_graph=False):

    grads = compute_grads(model, x_embeds, y_labels, lr=args.train_lr,ep=args.unlearn_ep, unlearn_md=args.unlearn_md,create_graph=create_graph)##############################
    return grad_dist(true_grads, grads, args)


def unlearn(model, x_embeds, y_labels,retain_set,lr=2e-5,ep=2,unlearn_md="ga", create_graph=False):
    #print(x_embeds, y_labels)
    ul_model=deepcopy(model)
    for e in range(ep):
        #print(x_embeds.shape,y_labels.shape)
        #exit()
        outs = ul_model(inputs_embeds=x_embeds, labels=y_labels)
        
        #print(outs.loss)
        #gd=torch.autograd.grad(-outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)
        if unlearn_md=="ga":
            #print(unlearn_md)
            gd=torch.autograd.grad((-1)*outs.loss, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        elif unlearn_md=="kl":
            #print(unlearn_md)
            with torch.no_grad():
                o_outs = model(inputs_embeds=x_embeds, labels=y_labels)
            gd=torch.autograd.grad((-1)*F.kl_div(outs.logits,o_outs.logits,reduction = 'batchmean',log_target = True), [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        elif unlearn_md=="npo":
            #print(unlearn_md)
            beta=0.1
            with torch.no_grad():
                o_outs = model(inputs_embeds=x_embeds, labels=y_labels)
            neg_log_ratio = o_outs.logits - outs.logits
            gd=torch.autograd.grad(-F.logsigmoid(beta * neg_log_ratio).mean() * 2 / beta, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        elif unlearn_md=="taskVec":
            #print(unlearn_md)
            gd=torch.autograd.grad(outs.loss, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        
        #gd=torch.autograd.grad(-outs.loss, [param for param in ul_model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
        #print(gd)
        for param, grad in zip(ul_model.parameters(), gd):
            if param.requires_grad and grad is not None:
                #print("yes")
                #exit()
                param.data -=grad#cancel the *lr for precision
    if unlearn_md=="taskVec":
        for param, o_param in zip(ul_model.parameters(), model.parameters()):
            param.data =o_param.data-(param.data-o_param.data)
    ul_model.eval()
    return ul_model



def compute_grads_autoreg(model, x_embeds, y_labels, create_graph=False):
    #print(x_embeds.shape,y_labels.shape)
    outs = model(inputs_embeds=x_embeds, labels=y_labels)
    return torch.autograd.grad(-outs.loss, [param for param in model.parameters() if param.requires_grad], create_graph=create_graph, allow_unused=True)
    #return torch.autograd.grad(-outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)

def grad_dist(grads1, grads2, args):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None) and g1.view(-1).norm(p=2)* g2.view(-1).norm(p=2) :
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2))
            elif args.loss == 'l2':
                ret += (g1 - g2).square().sum()
            else:
                assert False
            n_g += 1
            #print(ret)
    if args.loss == 'cos':
        ret /= n_g
    return ret



def get_closest_tokens_gpt(inputs_embeds, embeddings_weight, metric="cos"):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == "l2":
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == "cos":
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    return d, d.min(dim=2)[1]


def get_reconstruction_loss_autoreg(model, x_embeds, y_labels, true_grads, args, create_graph=False):
    grads = compute_grads_autoreg(model, x_embeds, y_labels, create_graph=create_graph)
    return grad_dist(true_grads, grads, args)




def fix_special_tokens(x_embeds, bert_embeddings_weight, pads):
    x_embeds.data[:, 0] = bert_embeddings_weight[BERT_CLS_TOKEN]
    if pads is not None:
        for sen_id in range(x_embeds.shape[0]):
            x_embeds.data[sen_id, pads[sen_id]:] = bert_embeddings_weight[BERT_PAD_TOKEN]
            x_embeds.data[sen_id, pads[sen_id]-1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    elif x_embeds.shape[0] == 1:
        x_embeds.data[:, -1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    return x_embeds


def remove_padding(tokenizer, ids):
    for i in range(ids.shape[0] - 1, -1, -1):
        if ids[i] == BERT_SEP_TOKEN:
            ids = ids[:i+1]
            break
    return tokenizer.decode(ids)


def clean_dataset(df, 
                  size: int = 25000):
    unique_specialties = df['medical_specialty'].dropna().unique()

    # 将结果转换为列表
    unique_specialties_list = unique_specialties.tolist()
    specialties_dict = {specialty: index for index, specialty in enumerate(unique_specialties_list)}
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()

    df_shuffled['text'] = df_shuffled['description']

    df_shuffled['label'] = df_shuffled['medical_specialty'].map(specialties_dict)
    
    return df_shuffled
class TextDataset_cls:
    def __init__(self, device, dataset_name, dataset_size, batch_size,split):
        self.size=dataset_size
        self.device=device
        self.dataset_name=dataset_name 
        self.batch_size=batch_size 
        self.split=split
        #np.random.seed(13)
        
        if dataset_name == "synthpai_age":
            if split=="train":
                df = pd.read_csv("synthpai_train_age.csv", index_col=False)
            else:
                df = pd.read_csv("synthpai_test_age.csv", index_col=False)

            dataset = Dataset.from_pandas(df)
        elif dataset_name == "synthpai_inc":
            if split=="train":
                df = pd.read_csv("synthpai_train_inc.csv", index_col=False)
            else:
                df = pd.read_csv("synthpai_test_inc.csv", index_col=False)

            dataset = Dataset.from_pandas(df)
        else:
            raise ValueError(f'This >{dataset_name}< is not provided, yet.')
        # Slice
        #print(dataset,type(dataset))
        #exit()
        idxs = list(range(dataset_size))
        self.seqs = []
        self.labels = []
        for i in range(dataset_size):
            seqs = []
            for j in range(batch_size):
                if i*batch_size+j<=len(idxs)-1:
                    seqs.append(dataset[idxs[i*batch_size+j]]["text"])
            labels = torch.tensor([dataset[idxs[i*batch_size : (i+1)*batch_size]]['label']], device=device)
            self.seqs.append(seqs)
            self.labels.append(labels)
        assert len(self.seqs) == dataset_size
        assert len(self.labels) == dataset_size


    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])
    
    def getlen(self):
        return self.size
    
    def get_subset(self, indices):
        subset = TextDataset_cls(self.device, self.dataset_name, len(indices), self.batch_size, self.split)
        subset.seqs = [self.seqs[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        return subset
    def mis_label(self, indices):
        #subset = TextDataset_cls(self.device, self.dataset_name, len(indices), self.batch_size, self.split)
        #subset.seqs = [self.seqs[i] for i in indices]
        #self.labels = [1-self.labels[i] for i in indices]
        for i in indices:
            self.labels[i]=1-self.labels[i]
    def shuffle(self):
        indices = np.random.permutation(self.size)
        self.seqs = [self.seqs[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
def fineTune_cls(model,tokenizer,dataset,test_dataset,lr=None,ep=3,bs=32):
    device=model.device

    
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 512
    length=dataset.getlen()
    if length<bs:
        bs=length
    num_batch=length//bs
  
    for e in range(ep):
        dataset=dataset.get_subset(range(dataset.getlen()))
        dataset.shuffle()

        step = 0
        train_loss = 0.
        total_sample = 0
        all_train_preds = []
        all_train_labels = []

        for i in range(0,length//bs):
            step += 1
            if i*bs+bs>length-1:
                seqs,labels=dataset[i*bs:length]
            else:
                seqs,labels=dataset[i*bs:i*bs+bs]

            labels=torch.tensor(labels).to(device)


            seqs=[seq[0] for seq in seqs]

            orig_batch = tokenizer(seqs, padding="max_length", truncation=True, max_length=12,return_tensors='pt').to(device)


            outs = model(input_ids=orig_batch['input_ids'], labels=labels)#此处取消了mask，因为假设敌手只知道最大长度，便于重建。如果加上mask，敌手那边也应当给mask，不然只传embedding和标签，重建效果会很差
            gd=torch.autograd.grad(outs.loss, [param for param in model.parameters() if param.requires_grad], create_graph=True, allow_unused=True)
            

            for param, grad in zip(model.parameters(), gd):
                if param.requires_grad and grad is not None:
                    #print("OK")
                    param.data -=grad*lr#cancel the *lr for precision

            train_loss += bs * outs.loss.item()
            total_sample += bs

            preds = torch.argmax(outs.logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            if step % 10== 0:
                train_loss = 0.
                total_sample = 0
                all_train_preds = []
                all_train_labels = []  
        model.eval()


    return model



