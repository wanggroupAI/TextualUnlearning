from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import torch
import numpy as np
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

        idxs = list(range(dataset_size))
        self.seqs = []
        self.labels = []
        for i in range(dataset_size):
            seqs = []
            for j in range(batch_size):
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