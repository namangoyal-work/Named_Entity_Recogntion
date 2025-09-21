import torch
import torch.nn as nn
import torch.optim as optim
from biodata import BioNERDataset, dl_collate_fn
from tqdm.auto import tqdm
import fasttext

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class BiLSTMNERTagger(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_layers = 1
        self.lstm = nn.LSTM(
                input_size=300, 
                batch_first=True, 
                hidden_size=300, 
                bidirectional=True,
                num_layers=self.num_layers
            )
        self.clf = nn.Linear(600, 13)
        self.h0 = nn.Parameter(torch.randn(2*self.num_layers,300))
        self.c0 = nn.Parameter(torch.randn(2*self.num_layers,300))
        
    def forward(self, x):

        batch_size = x.size(0)
        h = torch.stack([self.h0 for _ in range(batch_size)], dim=1)
        c = torch.stack([self.c0 for _ in range(batch_size)], dim=1)
        lstm_output, _ = self.lstm(x, (h,c))
        logits = self.clf(lstm_output)
        return logits

label2id = {'O': 0,
 'B-Biological_Molecule': 1,
 'E-Biological_Molecule': 2,
 'S-Biological_Molecule': 3,
 'I-Biological_Molecule': 4,
 'S-Species': 5,
 'B-Species': 6,
 'I-Species': 7,
 'E-Species': 8,
 'B-Chemical_Compound': 9,
 'E-Chemical_Compound': 10,
 'S-Chemical_Compound': 11,
 'I-Chemical_Compound': 12}

def preprocess_batch(ft_model, batch):
    
    sentences, labels = batch
    pt_sents = []
    pt_labels = []
    for s, l in zip(sentences, labels):
        sent = []
        for w in s:
            sent.append(torch.tensor(ft_model.get_word_vector(w), requires_grad=False))
        pt_sents.append(torch.vstack(sent))
        pt_labels.append(torch.tensor(l))
    
    return pad_sequence(pt_sents, batch_first=True), pad_sequence(pt_labels, batch_first=True)

if __name__ == "__main__":

    # train
    device = torch.device("mps")
    train_ds = BioNERDataset('train.txt', label2id)

    train_dl = DataLoader(train_ds, batch_size=8, collate_fn=dl_collate_fn, num_workers=2, shuffle=True)

    model = BiLSTMNERTagger().to(device)
    ft_model = fasttext.load_model('cc.en.300.bin')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):

        print(f'Epoch {epoch+1}:')
        train_loss = torch.tensor(0, dtype=torch.float, device=device)
        model.train()
        for batch in tqdm(train_dl):
            tokens, labels = preprocess_batch(ft_model, batch)
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(tokens).flatten(end_dim=1)
            loss = loss_fn(logits, labels.flatten())
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()

        train_loss = train_loss.cpu()
        train_loss /= len(train_dl)
        print(f' Train Loss: {train_loss}')

        # val_loss = 0
        # model.eval()
        # for batch in tqdm(val_dl):
        #     tok_batch = tokenize_and_align_labels(tokenizer, batch)
        #     tok_batch = {k : v.to(device) for k, v in tok_batch.items()}

        #     loss = model(**tok_batch).loss
        #     val_loss += loss.detach()

        # val_loss = val_loss.cpu()
        # val_loss /= len(val_dl)
        # print(f' Val Loss: {val_loss}')
        # print('')
