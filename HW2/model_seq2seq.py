import numpy as np
import pandas as pd
import random
import math
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import re
import pickle
import json
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-words', type=int, default=25)
    parser.add_argument('--n-input-step', type=int, default=80)
    parser.add_argument('--n-output-step', type=int, default=25)
    parser.add_argument('--hid-size', type=int, default=512)
    parser.add_argument('--feat-size', type=int, default=4096)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args, unknown = parser.parse_known_args()
    return args

class vocab():
    def __init__(self, caps):
        captions = []
        for cap_ in list(caps.values()):
            for cap in cap_:
                captions.append(cap)
        wc = {}
        for c in captions:
            w_ = c.split()
            for w in w_:
                if w not in wc:
                    wc[w] = 1
                else:
                    wc[w] += 1           
        self.vocabs = sorted([w for w, c in wc.items() if c >= 3])
        self.vocabs += ['<PAD>','<BOS>', '<EOS>', '<UNK>']
        v_size = len(self.vocabs)
        self.i_s = {i : self.vocabs[i] for i in range(v_size)}
        self.si = {self.vocabs[i] : i for i in range(v_size)}


class Vid2Cap(Dataset):  
    def __init__(self,l_file,f_file,vocab_=None):
        self.args = get_args()
        self.cap_pair = []
        self.id_to_feat = {}
        self.max_words = self.args.max_words
        cap_ = {}
        
        with open(l_file, 'r') as f:
            l_data = json.load(f)
        for i in range(len(l_data)):
            cap_[l_data[i]['id']] = [self.util(c) for c in l_data[i]['caption']]
        
        if vocab_ == None:
            self.vocab_ = vocab(cap_)
        else: 
            self.vocab_ = vocab_
        
        for id, c_ in cap_.items():
            self.id_to_feat[id] = torch.FloatTensor(np.load(f_file+id+'.npy'))
            for c in c_: 
                Cap = ['<BOS>']
                for w in c.split():
                    if w in self.vocab_.vocabs:
                        Cap.append(w)
                    else:
                        Cap.append('<UNK>')
                if len(Cap)+1 > self.max_words:
                    continue
                Cap += ['<EOS>']    
                Cap += ['<PAD>']*(self.max_words - len(Cap))
                Cap = ' '.join(Cap)
                self.cap_pair.append((id, Cap)) 

    def util(self,i):
        i = i.lower().strip()
        i = re.sub(r'([.,?!])', r' \1 ', i)
        i = re.sub(r'[" "]+', r' ', i)
        i = re.sub(r'[^A-Za-z.,?!]+', r' ', i)
        i = i.strip()
        return i
    
    def __getitem__(self,idx):
        id, c = self.cap_pair[idx]
        f = self.id_to_feat[id]
        w = torch.LongTensor([self.vocab_.si[i] for i in c.split(' ')])
        c_ = torch.LongTensor(self.max_words, len(self.vocab_.vocabs))
        c_.zero_()
        c_.scatter_(1, w.view(self.max_words, 1), 1)
        return {'feature': f, 'one_hot': c_, 'caption': c}

    def __len__(self):
        return len(self.cap_pair)

class Att_(nn.Module):
    def __init__(self, h_size, d=0.5):
        super(Att_,self).__init__()
        self.Attention = nn.Linear(h_size*2, 1)
        self.dropout = nn.Dropout(p=d)
        self.hidden_size = h_size 
        
    def forward(self, h_, output):
        out = torch.bmm(output.transpose(0,1),h_.transpose(0,1).transpose(1,2))
        out = torch.nn.functional.tanh(out)
        w = torch.nn.functional.softmax(out,dim=1)
        return w
        
class SEQ2SEQ(nn.Module):
    def __init__(self, vocab, i, o, f_size, h_size, e_size, n=1, d=0.5):
        super(SEQ2SEQ, self).__init__()
        self.args = get_args()
        self.vocab = vocab
        self.i = i
        self.o = o
        self.f_size = f_size
        self.h_size = h_size
        self.e_size = e_size
        self.f_size_ = e_size 
        
        self.fc        = nn.Linear(f_size,self.f_size_)
        self.dropout   = nn.Dropout(p=d)
        self.encoder   = nn.GRU(self.f_size_,h_size,n)
        self.attention = Att_(h_size)
        self.decoder   = nn.GRU(h_size*2+e_size,h_size,n)
        v_size         = len(self.vocab.vocabs)
        self.output    = nn.Linear(h_size, v_size)
        self.softmax   = nn.LogSoftmax(dim=1)
        self.embed     = nn.Embedding(v_size,e_size)
    
    def forward(self, i, o, tfr=-1):
        b_size = i.shape[1]
        e_pad = Variable(torch.zeros(self.o,b_size,self.f_size_)).to(self.args.device)
        d_pad = Variable(torch.zeros(self.i,b_size,self.h_size+self.e_size)).to(self.args.device) 
        i = self.dropout(torch.nn.functional.leaky_relu(self.fc(i)))         
        e = torch.cat((i,e_pad),0)
        e_, _ = self.encoder(e) 
        decoder_one = torch.cat((d_pad,e_[:self.i,:,:]),2)
        decoder_one_,alpha = self.decoder(decoder_one) 
        cap = self.embed(o)        
        b = [self.vocab.si['<BOS>']]*b_size
        b = Variable(torch.LongTensor([b])).resize(b_size,1).to(self.args.device)

        loss = 0.0
        for j in range(self.o):
            if j==0:
                d = self.embed(b)
            elif random.random()<=tfr:
                d = cap[:,j-1,:].unsqueeze(1)
            else:
                d = self.embed(decoder_two_.max(1)[-1].resize(b_size,1))
            w = self.attention(alpha,e_[:self.i])
            c = torch.bmm(w.transpose(1,2),e_[:self.i].transpose(0,1))
            decoder_two = torch.cat((d, e_[self.i+j].unsqueeze(1),c),2).transpose(0,1)
            decoder_two_,alpha = self.decoder(decoder_two,alpha)
            decoder_two_ = self.softmax(self.output(decoder_two_[0]))
            loss += torch.nn.functional.nll_loss(decoder_two_,o[:,j])/self.o
        return loss
    
    def pred(self, input, bw=1):
        e_pad = Variable(torch.zeros(self.o,1,self.f_size_)).to(self.args.device)
        d_pad = Variable(torch.zeros(self.i,1,self.h_size+self.e_size)).to(self.args.device) 
        input = torch.nn.functional.leaky_relu(self.fc(input))        
        e = torch.cat((input, e_pad),0)
        e_,_ = self.encoder(e)
        decoder_one = torch.cat((d_pad, e_[:self.i,:,:]),2)
        decoder_one_, alpha = self.decoder(decoder_one)
        b = [self.vocab.si['<BOS>']]
        b = Variable(torch.LongTensor([b])).resize(1,1).to(self.args.device)
        
        if bw>1:
            for i in range(self.o):
                if i == 0: 
                    d = self.embed(b)
                    w = self.attention(alpha,e_[:self.i])
                    c = torch.bmm(w.transpose(1,2),e_[:self.i].transpose(0,1))
                    decoder_two = torch.cat((d,e_[self.i+i].unsqueeze(1),c),2).transpose(0,1)
                    d_,alpha = self.decoder(decoder_two,alpha)
                    d_ = self.softmax(self.output(d_[0]))
                    prob = math.e**d_
                    top_k, idx = prob.topk(bw)
                    scores = top_k.data[0].cpu().numpy().tolist()
                    all = idx.data[0].cpu().numpy().reshape(bw,1).tolist()
                    alpha_ = [alpha] * bw
                else:
                    n_=[]
                    for j, z in enumerate(all):
                        d = Variable(torch.LongTensor([z[-1]])).to(self.args.device).resize(1,1)
                        d = self.embed(d)
                        w = self.attention(alpha,e_[:self.i])
                        c = torch.bmm(w.transpose(1,2),e_[:self.i].transpose(0,1))
                        decoder_two = torch.cat((d,e_[self.i+i].unsqueeze(1),c),2).transpose(0,1)
                        d_,alpha_[j] = self.decoder(decoder_two,alpha_[j])
                        d_ = self.softmax(self.output(d_[0]))
                        prob = math.e**d_
                        top_k, idx = prob.topk(bw)
                        for k in range(bw):
                            s = scores[j]*top_k.data[0, k]
                            n = all[j]+[idx[0,k].item()] 
                            n_.append([s,n,alpha_[j]])
                    n_ = sorted(n_,key=lambda y: y[0], reverse=True)[:bw]
                    scores = [z[0] for z in n_]
                    all = [z[1] for z in n_]
                    alpha_ = [all[2] for all in n_]
            token_idx = [self.vocab.si[t] for t in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']]
            output = [self.vocab.i_s[int(l)] for l in all[0] if int(l) not in token_idx]
            return output
        else:
            p=[]
            for i in range(self.o):
                if i == 0: 
                    d = self.embed(b)
                else:
                    d = self.embed(d_.max(1)[-1].resize(1,1))
                w = self.attention(alpha,e_[:self.i])
                c = torch.bmm(w.transpose(1,2),e_[:self.i].transpose(0,1))
                decoder_two = torch.cat((d,e_[self.i+i].unsqueeze(1),c),2).transpose(0,1)
                d_,alpha = self.decoder(decoder_two,alpha)
                d_ = self.softmax(self.out(d_[0]))
                token_idx = [self.vocab.si[q] for q in ['<EOS>', '<PAD>', '<UNK>']]
                id = d_.max(1)[-1].item()
                if id in token_idx:
                    break
                elif id != self.vocab.si['<BOS>']:
                    p.append(self.vocab.i_s[int(id)])
            return p

def Train(train_loader, model, Epoch=50, lr=1e-3):
    args = get_args()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    Loss = []
    for e in range(Epoch):
        model.train()
        epoch_loss = 0.0
        for i, data in tqdm(enumerate(train_loader)): 
            x = data['feature'].transpose(0,1).to(args.device)
            y = data['one_hot'].to(args.device)
            optimizer.zero_grad()
            loss = model(x,y.max(2)[-1],tfr=0.05)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / len(train_loader)
        Loss.append(epoch_loss)
        torch.save(model.state_dict(),'Model')
        print('Epoch {}, Loss = {:.2f}'.format(e,epoch_loss))
    return Loss

def Test(model, vocab_file, test_feature, test_id_file, output_file):
    args = get_args()
    vocab = pickle.load(open(vocab_file, "rb"))
    S = SEQ2SEQ(vocab, args.n_input_step, args.n_output_step, args.feat_size, args.hid_size, args.embed_size).to(args.device)
    S.load_state_dict(torch.load(model))
    S.to(args.device)
    d = {}
    l = pd.read_fwf(test_id_file, header=None)
    for _, i in l.iterrows():
        f = f"{test_feature}{i[0]}.npy"
        d[i[0]] = torch.FloatTensor(np.load(f))
    S.eval()
    pred = []
    idx = []
    for _, i in l.iterrows():
        j = Variable(d[i[0]].view(-1,1,args.feat_size)).to(args.device)
        p = S.pred(j,bw=2)
        p = " ".join(p).capitalize().replace(' .', "")
        pred.append(p)
        idx.append(i[0])
    with open(output_file, 'w') as file:
        for i, _ in l.iterrows():
            file.write(idx[i] + "," + pred[i] + "\n")

#################################################################
# Train
#################################################################
'''
print('Train ...')
args = get_args()
trainset = Vid2Cap('./data/training_label.json', './data/training_data/feat/')
train_loader = DataLoader(trainset,batch_size=64)
pickle.dump(trainset.vocab_, open('vocab.pickle', 'wb'))
S = SEQ2SEQ(trainset.vocab_, args.n_input_step, args.n_output_step, args.feat_size, args.hid_size, args.embed_size).to(args.device)
train_loss = Train(train_loader, S)
'''
#################################################################
# Test
#################################################################
print('Test ...')
Test('Model', 'vocab.pickle', './data/testing_data/feat/', './data/testing_data/id.txt', 'output.txt')