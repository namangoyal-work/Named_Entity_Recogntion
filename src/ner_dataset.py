import torch
from torch.utils.data import Dataset

class BioNERDataset(Dataset):
    
    def __init__(self, filename, label2id):
        
        self.sentences = []
        self.token_counts = 0
        self.class_counts = torch.zeros(len(label2id))
        with open(filename, 'r') as f:
            sentence = []
            for l in f:
                if l == '\n':
                    self.sentences.append(list(zip(*sentence)))
                    sentence = []
                else:
                    token, cls = l.strip().split('\t')
                    sentence.append((token, label2id[cls]))
                    self.class_counts[label2id[cls]] += 1
                    self.token_counts += 1
        
        self.sentences.append(list(zip(*sentence)))
        self.weights = self.token_counts / (self.class_counts*len(label2id))
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx]

def dl_collate_fn(data):
    return tuple(zip(*data))
