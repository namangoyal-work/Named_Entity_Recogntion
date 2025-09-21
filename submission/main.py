import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import random
import sys

from torchtext.vocab import GloVe

from torch.nn.utils.rnn import pad_sequence
import copy
import numpy as np
from sklearn.metrics import f1_score

input_dir = '/kaggle/input'
data_path = f'{input_dir}/col772-a2-data'

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
# if torch.backends.mps.is_available():
#     device = torch.device('mps')

print(device)

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

class BioNEREvalDataset(Dataset):
    
    def __init__(self, filename):
        
        self.sentences = []
        with open(filename, 'r') as f:
            sentence = []
            for l in f:
                if l == '\n':
                    self.sentences.append(sentence)
                    sentence = []
                else:
                    token = l.strip()
                    sentence.append(token)
        
        self.sentences.append(sentence)
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx]

def dl_collate_fn(data):
    return tuple(zip(*data))

def eval_dl_collate_fn(data):
    return tuple(data)

class GloveModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        glove = GloVe('6B')
                    
        # https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        unk = torch.tensor([0.22418134, -0.28881392, 0.13854356, 0.00365387, 
                    -0.12870757, 0.10243822, 0.061626635, 0.07318011, 
                    -0.061350107, -1.3477012, 0.42037755, -0.063593924, 
                    -0.09683349, 0.18086134, 0.23704372, 0.014126852, 
                    0.170096, -1.1491593, 0.31497982, 0.06622181, 
                    0.024687296, 0.076693475, 0.13851812, 0.021302193, 
                    -0.06640582, -0.010336159, 0.13523154, -0.042144544, 
                    -0.11938788, 0.006948221, 0.13333307, -0.18276379, 
                    0.052385733, 0.008943111, -0.23957317, 0.08500333, 
                    -0.006894406, 0.0015864656, 0.063391194, 0.19177166, 
                    -0.13113557, -0.11295479, -0.14276934, 0.03413971, 
                    -0.034278486, -0.051366422, 0.18891625, -0.16673574, 
                    -0.057783455, 0.036823478, 0.08078679, 0.022949161, 
                    0.033298038, 0.011784158, 0.05643189, -0.042776518, 
                    0.011959623, 0.011552498, -0.0007971594, 0.11300405, 
                    -0.031369694, -0.0061559738, -0.009043574, -0.415336, 
                    -0.18870236, 0.13708843, 0.005911723, -0.113035575, 
                    -0.030096142, -0.23908928, -0.05354085, -0.044904727, 
                    -0.20228513, 0.0065645403, -0.09578946, -0.07391877, 
                    -0.06487607, 0.111740574, -0.048649278, -0.16565254, 
                    -0.052037314, -0.078968436, 0.13684988, 0.0757494, 
                    -0.006275573, 0.28693774, 0.52017444, -0.0877165, 
                    -0.33010918, -0.1359622, 0.114895485, -0.09744406, 
                    0.06269521, 0.12118575, -0.08026362, 0.35256687, 
                    -0.060017522, -0.04889904, -0.06828978, 0.088740796, 
                    0.003964443, -0.0766291, 0.1263925, 0.07809314, 
                    -0.023164088, -0.5680669, -0.037892066, -0.1350967, 
                    -0.11351585, -0.111434504, -0.0905027, 0.25174105, 
                    -0.14841858, 0.034635577, -0.07334565, 0.06320108, 
                    -0.038343467, -0.05413284, 0.042197507, -0.090380974, 
                    -0.070528865, -0.009174437, 0.009069661, 0.1405178, 
                    0.02958134, -0.036431845, -0.08625681, 0.042951006, 
                    0.08230793, 0.0903314, -0.12279937, -0.013899368, 
                    0.048119213, 0.08678239, -0.14450377, -0.04424887, 
                    0.018319942, 0.015026873, -0.100526, 0.06021201, 
                    0.74059093, -0.0016333034, -0.24960588, -0.023739101, 
                    0.016396184, 0.11928964, 0.13950661, -0.031624354, 
                    -0.01645025, 0.14079992, -0.0002824564, -0.08052984, 
                    -0.0021310581, -0.025350995, 0.086938225, 0.14308536, 
                    0.17146006, -0.13943303, 0.048792403, 0.09274929, 
                    -0.053167373, 0.031103406, 0.012354865, 0.21057427, 
                    0.32618305, 0.18015954, -0.15881181, 0.15322933, 
                    -0.22558987, -0.04200665, 0.0084689725, 0.038156632, 
                    0.15188617, 0.13274793, 0.113756925, -0.095273495, 
                    -0.049490947, -0.10265804, -0.27064866, -0.034567792, 
                    -0.018810693, -0.0010360252, 0.10340131, 0.13883452, 
                    0.21131058, -0.01981019, 0.1833468, -0.10751636, 
                    -0.03128868, 0.02518242, 0.23232952, 0.042052146, 
                    0.11731903, -0.15506615, 0.0063580726, -0.15429358, 
                    0.1511722, 0.12745973, 0.2576985, -0.25486213, 
                    -0.0709463, 0.17983761, 0.054027, -0.09884228, 
                    -0.24595179, -0.093028545, -0.028203879, 0.094398156, 
                    0.09233813, 0.029291354, 0.13110267, 0.15682974, 
                    -0.016919162, 0.23927948, -0.1343307, -0.22422817, 
                    0.14634751, -0.064993896, 0.4703685, -0.027190214, 
                    0.06224946, -0.091360025, 0.21490277, -0.19562101, 
                    -0.10032754, -0.09056772, -0.06203493, -0.18876675, 
                    -0.10963594, -0.27734384, 0.12616494, -0.02217992, 
                    -0.16058226, -0.080475815, 0.026953284, 0.110732645, 
                    0.014894041, 0.09416802, 0.14299914, -0.1594008, 
                    -0.066080004, -0.007995227, -0.11668856, -0.13081996, 
                    -0.09237365, 0.14741232, 0.09180138, 0.081735, 
                    0.3211204, -0.0036552632, -0.047030564, -0.02311798, 
                    0.048961394, 0.08669574, -0.06766279, -0.50028914, 
                    -0.048515294, 0.14144728, -0.032994404, -0.11954345, 
                    -0.14929578, -0.2388355, -0.019883996, -0.15917352, 
                    -0.052084364, 0.2801028, -0.0029121689, -0.054581646, 
                    -0.47385484, 0.17112483, -0.12066923, -0.042173345, 
                    0.1395337, 0.26115036, 0.012869649, 0.009291686, 
                    -0.0026459037, -0.075331464, 0.017840583, -0.26869613, 
                    -0.21820338, -0.17084768, -0.1022808, -0.055290595, 
                    0.13513643, 0.12362477, -0.10980586, 0.13980341, 
                    -0.20233242, 0.08813751, 0.3849736, -0.10653763, 
                    -0.06199595, 0.028849555, 0.03230154, 0.023856193, 
                    0.069950655, 0.19310954, -0.077677034, -0.144811])
    
        self.emb_dict = glove.stoi
        self.unk_index = 400000
        self.embedding = nn.Embedding.from_pretrained(torch.vstack([glove.vectors,unk]))
    
    def forward(self, sentence):
        sentence = torch.tensor([self.emb_dict[word.lower()] if word.lower() in self.emb_dict else self.unk_index for word in sentence], device=device, dtype=torch.int32)
        return self.embedding(sentence)

class LSTMByteEmbeddingModel(nn.Module):
    
    def __init__(self, emb_dim=50):
        super().__init__()
        self.char_embeds  = nn.Embedding(256,emb_dim)
        self.lstm = nn.LSTM(
                input_size=emb_dim,
                batch_first=True, 
                hidden_size=emb_dim,
                bidirectional=True,
            )
        self.h0 = nn.Parameter(torch.randn(2,emb_dim))
        self.c0 = nn.Parameter(torch.randn(2,emb_dim))
    
    def forward(self, sentence):
        
        x = pad_sequence([torch.tensor(bytearray(w.encode('utf-8'))).to(device) for w in sentence], batch_first=True)
        x = self.char_embeds(x)
        
        batch_size = len(sentence)
        h0 = torch.stack([self.h0 for _ in range(batch_size)], dim=1)
        c0 = torch.stack([self.c0 for _ in range(batch_size)], dim=1)
        _, (h, c) = self.lstm(x, (h0, c0))
        return torch.cat((c[0], c[1]), 1)

class EmbeddingModel(nn.Module):

    # Embedding: 400
    #  - Pretrained: 300
    #  - Character: 100
    #    - L: 50
    #    - R: 50
    #
    # Tentative:
    #  - POS: 25
    #  - Chunk: 10
    #  - Dict:  5 (may not be allowed)
    def __init__(self):
        super().__init__()
        self.byte_emb_model = LSTMByteEmbeddingModel()            
        self.word_emb_model = GloveModel()
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, sentence):
        word_emb = self.word_emb_model(sentence)
        byte_emb = self.byte_emb_model(sentence)        
        return self.dropout(torch.cat([word_emb, byte_emb], 1))

class ProjectiveAttention(nn.Module):
    
    def __init__(self, emb_dim = 800):
        super().__init__()
        self.W = nn.Parameter(torch.zeros((emb_dim, emb_dim)))
        self.dropout = nn.Dropout(p=0.2)
        nn.init.normal_(self.W.data, 0, 0.1)
        
    def forward(self, h):
        # h: (B, N, K)
        
        # att = softmax(h@W@h_T, axis=1) @ h, across all batches
        att = torch.bmm(
                F.softmax(
                    torch.bmm(
                        h, 
                        torch.bmm(
                            self.W.unsqueeze(0).expand([h.size(0),-1,-1]), 
                            h.transpose(2,1)
                        )
                    ), dim=2), 
                h
              )
        return self.dropout(att)

class PostAttentionLayer(nn.Module):
    
    def __init__(self, emb_dim = 1600, proj_dim = 800):
        super().__init__()
        self.fwd = nn.Linear(emb_dim, proj_dim, bias=False)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, g, h):
        return self.dropout(torch.tanh(self.fwd(torch.cat([g,h], dim=2))))

class BiLSTMNERTagger(nn.Module):

    def __init__(self, num_layers=2, embedding_size=400):
        super().__init__()

        self.num_layers = num_layers
        self.lstm = nn.LSTM(
                input_size=embedding_size, 
                batch_first=True, 
                hidden_size=embedding_size, 
                bidirectional=True,
                num_layers=self.num_layers,
                dropout=0.2
            )
        self.pa = ProjectiveAttention(emb_dim=embedding_size*2)
        self.pal = PostAttentionLayer(emb_dim=embedding_size*4, proj_dim = embedding_size*2)
        self.clf = nn.Linear(2*embedding_size, 13)
        self.lstm_final_dropout = nn.Dropout(p=0.2)

        self.h0 = nn.Parameter(torch.zeros(2*self.num_layers,embedding_size))
        self.c0 = nn.Parameter(torch.zeros(2*self.num_layers,embedding_size))
        nn.init.uniform_(self.h0.data, 0, 0.1)
        nn.init.uniform_(self.c0.data, 0, 0.1)
        
    def forward(self, x):

        batch_size = x.size(0)
        h = torch.stack([self.h0 for _ in range(batch_size)], dim=1)
        c = torch.stack([self.c0 for _ in range(batch_size)], dim=1)
        lstm_output, _ = self.lstm(x, (h,c))
        lstm_dropout_output = self.lstm_final_dropout(lstm_output)
        logits = self.clf(self.pal(lstm_dropout_output, self.pa(lstm_dropout_output)))
        return logits

def preprocess_batch(emb_model, sentences, labels=None):
    
    pt_sents = []
    pt_labels = []
    pt_masks = []
    i = 0
    for i in range(len(sentences)):
        emb = emb_model(sentences[i])
        pt_sents.append(emb)
        if labels:
            pt_labels.append(torch.tensor(labels[i]))
        pt_masks.append(torch.tensor([1]*emb.shape[0]))
        
        i += 1
        
    if labels:
        return pad_sequence(pt_sents, batch_first=True), pad_sequence(pt_labels, batch_first=True, padding_value=-100), (pad_sequence(pt_masks, batch_first=True) > 0)
    
    return pad_sequence(pt_sents, batch_first=True), (pad_sequence(pt_masks, batch_first=True) > 0)
    
label2id = {'O': 0,
 'B-Biological_Molecule': 1,
 'E-Biological_Molecule': 2,
 'S-Biological_Molecule': 3,
 'I-Biological_Molecule': 4,
 'B-Species': 5,
 'E-Species': 6,
 'S-Species': 7,
 'I-Species': 8,
 'B-Chemical_Compound': 9,
 'E-Chemical_Compound': 10,
 'S-Chemical_Compound': 11,
 'I-Chemical_Compound': 12}

id2label = {b:a for a,b in label2id.items()}

def seed(seed=3143):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    seed()

    if (sys.argv[1] == 'train'):

        train_file_path = sys.argv[2]
        val_file_path = sys.argv[3]
        train_ds = BioNERDataset(train_file_path, label2id)
        val_ds = BioNERDataset(val_file_path, label2id)

        train_dl = DataLoader(train_ds, batch_size=16, collate_fn=dl_collate_fn, num_workers=8, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=16, collate_fn=dl_collate_fn, num_workers=8, shuffle=False)

        model = BiLSTMNERTagger(num_layers=2).to(device)

        emb_model = EmbeddingModel().to(device)

        optimizer = optim.Adam(list(model.parameters())+list(emb_model.parameters()), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        loss_fn = nn.CrossEntropyLoss(weight=train_ds.weights.to(device))

        best_model = None
        best_emb_model = None
        f1_means = []
        best_val_loss = 10000
        best_f1_mean = 0
        val_losses = []
        train_losses = []
        possible_labels = range(1,13)
        patience = 0

        for epoch in range(30):

            print(f'Epoch {epoch+1}:')
            train_loss = torch.tensor(0, dtype=torch.float, device=device)
            model.train()
            emb_model.train()
            for (sentences, answers) in tqdm(train_dl):
                tokens, labels, masks = preprocess_batch(emb_model, sentences, labels=answers)
                tokens = tokens.to(device)
                labels = labels.to(device).flatten()

                optimizer.zero_grad()
                logits = model(tokens).flatten(end_dim=1)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach()
            
            scheduler.step()

            train_loss = train_loss.cpu()
            train_loss /= len(train_dl)
            print(f' Train Loss: {train_loss}')
            train_losses.append(train_loss)

            val_loss = torch.tensor(0, dtype=torch.float, device=device)
            true_labels = []
            pred_labels = []
            model.eval()
            emb_model.eval()
            for (sentences, answers) in tqdm(val_dl):
                tokens, labels, mask = preprocess_batch(emb_model, sentences, labels=answers)
                tokens = tokens.to(device)
                labels = labels.to(device).flatten()
                mask = mask.to(device).flatten()

                logits = model(tokens).flatten(end_dim=1)
                loss = loss_fn(logits, labels)

                val_loss += loss.detach()
                
                pred_labels.append(torch.argmax(logits, axis=1)[mask])
                true_labels.append(labels[mask])
            
            val_loss = val_loss.cpu()
            val_loss /= len(val_dl)
            val_losses.append(val_loss)
            
            pred_labels = np.array(torch.hstack(pred_labels).cpu())
            true_labels = np.array(torch.hstack(true_labels).cpu())

            f1_micro = f1_score(true_labels, pred_labels, average="micro", labels=possible_labels)
            f1_macro = f1_score(true_labels, pred_labels, average="macro", labels=possible_labels)

            f1_mean = (f1_macro + f1_micro)/2
            f1_means.append(f1_mean)

            print(f' Val Loss: {val_loss}')
            print(f' F1 micro: {f1_micro}')
            print(f' F1 macro: {f1_macro}')
            print(f' F1 mean: {f1_mean}')
            print('')
            
            # early stopping
            if f1_mean <= best_f1_mean:
                if patience >= 4:
                    break
                else:
                    patience += 1
            else:
                patience = 0
                best_val_loss = val_loss
                best_f1_mean = f1_mean
                best_model = copy.deepcopy(model)
                best_emb_model = copy.deepcopy(emb_model)

        print(f'Best F1 mean: {best_f1_mean}')

        torch.save(best_model, 'cs1200869_model.pt')
        torch.save(best_emb_model, 'cs1200869_emb_model.pt')

    elif sys.argv[1] == 'test':
        test_file_path = sys.argv[2]
        output_file_path = sys.argv[3]
        eval_ds = BioNEREvalDataset(test_file_path)

        eval_dl = DataLoader(eval_ds, batch_size=32, collate_fn=eval_dl_collate_fn, num_workers=2, shuffle=False)

        # assuming the models have already been generated and are in the same directory
        model = torch.load('cs1200869_model.pt', map_location=device)
        emb_model = torch.load('cs1200869_emb_model.pt', map_location=device)

        model.eval()
        emb_model.eval()
        outputs = []
        for batch in tqdm(eval_dl):
            tokens, mask = preprocess_batch(emb_model, batch)
            tokens = tokens.to(device)
            mask = mask.to(device)

            logits = model(tokens)
            preds = torch.argmax(logits, dim=2)

            preds = [preds[i][mask[i]] for i in range(preds.shape[0])]
            preds_str = [[id2label[i.item()] for i in s] for s in preds]
            
            outputs += [list(zip(batch[i], preds_str[i])) for i in range(len(batch))]

        with open(output_file_path, 'w') as outfile:
            for s in outputs:
                for t in s:
                    outfile.write(f'{t[0]}\t{t[1]}\n')
                outfile.write('\n')
