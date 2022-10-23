import pandas as pd
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from functools import partial

class SEMIDataset(Dataset):
    def __init__(self, sents, sents_aug1, sents_aug2, labels=None):
        self.sents = sents
        self.sents_aug1 = sents_aug1
        self.sents_aug2 = sents_aug2
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.sents_aug1[idx], self.sents_aug2[idx], self.labels[idx]

class SEMINoAugDataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

class MyCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sents, sents_aug1, sents_aug2 = [], [], []
        labels = []
        for sample in batch:
            if len(sample) == 2:
                sents.append(sample[0])
                labels.append(sample[1])
                sents_aug1 = None
                sents_aug2 = None
            elif len(sample) == 4:
                sents.append(sample[0])
                sents_aug1.append(sample[1])
                sents_aug2.append(sample[2])
                labels.append(sample[3])
    
        tokenized = self.tokenizer(sents, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        labels = torch.LongTensor(labels) - 1
        if sents_aug1 is not None:
            tokenized_aug1 = self.tokenizer(sents_aug1, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        else:
            tokenized_aug1 = None
        if sents_aug2 is not None: 
            tokenized_aug2 = self.tokenizer(sents_aug2, padding=True, truncation='longest_first', max_length=255, return_tensors='pt')
        else:
            tokenized_aug2 = None
        return tokenized, tokenized_aug1, tokenized_aug2, labels

def get_dataloader(data_path, labeled_size=200, mu=4, load_mode='semi'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_l_df = pd.read_csv(os.path.join(data_path,'train_{}.csv'.format(labeled_size)))
    dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
    
    
    if load_mode == 'semi':
        train_u_df = pd.read_csv(os.path.join(data_path,'unlabeled_data.csv'))
        if 'yahoo' in data_path:
            bt_l_df = pd.read_csv(os.path.join(data_path, 'bt_{}.csv'.format(labeled_size)))
            bt_u_df = pd.read_csv(os.path.join(data_path, 'bt_unlabeled.csv'.format(labeled_size)))
            train_dataset_l = SEMIDataset(train_l_df['content'].to_list(), train_l_df['synonym_aug'].to_list(), bt_l_df['back_translation'], labels=train_l_df['label'].to_list())
            train_dataset_u = SEMIDataset(train_u_df['content'].to_list(), train_u_df['synonym_aug'].to_list(), bt_u_df['back_translation'], labels=train_u_df['label'].to_list())
        else:
            train_dataset_l = SEMIDataset(train_l_df['content'].to_list(), train_l_df['synonym_aug'].to_list(), train_l_df['back_translation'], labels=train_l_df['label'].to_list())
            train_dataset_u = SEMIDataset(train_u_df['content'].to_list(), train_u_df['synonym_aug'].to_list(), train_u_df['back_translation'], labels=train_u_df['label'].to_list())
        train_loader_u = DataLoader(dataset=train_dataset_u, batch_size=32, shuffle=True, collate_fn=MyCollator(tokenizer))
    
    elif load_mode == 'baseline':
        train_dataset_l = SEMINoAugDataset(train_l_df['content'].to_list(), train_l_df['label'].to_list())
        train_loader_u = None
        
    dev_dataset = SEMINoAugDataset(dev_df['content'].to_list(), labels=dev_df['label'].to_list())
    test_dataset = SEMINoAugDataset(test_df['content'].to_list(), labels=test_df['label'].to_list())

    train_loader_l = DataLoader(dataset=train_dataset_l, batch_size=32, shuffle=True, collate_fn=MyCollator(tokenizer))
    
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=False, collate_fn=MyCollator(tokenizer))
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, collate_fn=MyCollator(tokenizer))
    
    num_class = max(train_l_df['label'].to_list())
    return train_loader_l, train_loader_u, dev_loader, test_loader, num_class