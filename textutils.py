import torch.nn as nn
import torch
import time
import numpy as np


## models

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)
    
    
    
## utils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def train_fl(models, iterators, optimizers, criterion, datanumbers, usersvec, iter_iters=None):
    
    epoch_loss = 0
    epoch_acc = 0
    
    [m.train() for m in models]
    
    cnt = 0
    for mi, (itr, usel) in enumerate(zip(iterators, usersvec)):
        if not usel:
            continue
        if iter_iters:
            n = len(itr)
            i_ex = np.random.permutation(n)[:iter_iters]
            
        for bi, batch in enumerate(itr):
            if iter_iters and bi not in i_ex:
                continue
                
            optimizers[mi].zero_grad()

            text, text_lengths = batch.text

            predictions = models[mi](text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            nn.utils.clip_grad_norm_(models[mi].parameters(), 5)
            optimizers[mi].step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            cnt += 1
                
    return epoch_loss / cnt, epoch_acc / cnt


def train_fl2(models, iterators, optimizers, criterion, datanumbers, usersvec):
    
    epoch_loss = 0
    epoch_acc = 0
    
    [m.train() for m in models]
    
    cnt = 0
    for mi, (itr, usel) in enumerate(zip(iterators, usersvec)):
        if not usel:
            continue
        for bi, batch in enumerate(itr):
            if bi > 10:
                continue
            optimizers[mi].zero_grad()

            text, text_lengths = batch.text

            predictions = models[mi](text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            nn.utils.clip_grad_norm_(models[mi].parameters(), 1)
            optimizers[mi].step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            cnt += 1
                
    return epoch_loss / cnt, epoch_acc / cnt



def aggregate_global_model(usvec, kweights, lnets, gnet):
    # number of users joining the curr iter
    finalb = np.where(usvec)[0]
    n_users = len(finalb)
    aflag = False

    # average all users parameters  
    if n_users > 0:                
        state_dict_new = lnets[finalb[0]].state_dict()
        for jj,fj in enumerate(finalb):
            state_dict_curr = lnets[fj].state_dict()
            for key in state_dict_new:
                if jj == 0:
                    state_dict_new[key] *= kweights[fj] 
                elif jj < n_users-1:  
                    state_dict_new[key] += state_dict_curr[key] * kweights[fj]
                else:
                    state_dict_new[key] /= kweights[finalb].sum()

        #initialize these matirces used for global FL model update
        gnet.load_state_dict(state_dict_new) 
        aflag = True
        
        #print(finalb)
        
    return gnet, aflag


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs