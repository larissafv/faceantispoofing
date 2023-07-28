import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch import nn, Tensor
import os
import time
import math

seed = 123
rng = np.random.default_rng(seed)

import model as m
import dataset as d

parser = argparse.ArgumentParser(add_help = True, description = 'Face Anti-Spoofing')
parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size')
parser.add_argument('--epochs', type = int, default = 50, help = 'Number of epochs')
parser.add_argument('--patience', type = int, default = 7, help = 'Number of epochs without validation loss decrease')
parser.add_argument('--data_folder', type = str, default = './img_data/', help = 'Folder location with all data')
parser.add_argument('--load_model', type = str, default = './models/', help='Path to saved model')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Getting dataset...')
train_data = pd.read_csv(args.data_folder + 'train.csv')
val_data   = pd.read_csv(args.data_folder + 'val.csv')
test_data  = pd.read_csv(args.data_folder + 'test.csv')

train_dataset = d.IMG_Dataset(train_data['path'], train_data['live'], args.batch_size, rng, device)
val_dataset   = d.IMG_Dataset(val_data['path'], val_data['live'], args.batch_size, rng, device)
test_dataset  = d.IMG_Dataset(test_data['path'], test_data['live'], 1, rng, device)

print('Creating model...')
model = m.Face_AntiSpoofing().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

def train(model):
    model.train()
    total_loss = 0.
    parcial_loss = 0.
    log_interval = 200
    start_time = time.time()
    
    n_batches = train_data.shape[0] // args.batch_size
    aux_train = list(range(train_data.shape[0]))
    for n in range(n_batches):
        batch, x, y = train_dataset[aux_train]
        aux_train = [x for x in aux_train if x not in batch]
        
        output = model(x)
        loss = criterion(output.view(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        parcial_loss += loss.item()
        if n % log_interval == 0 and n > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = parcial_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {n:5d}/{n_batches:5d} batches | '
                  f'lr {args.lr:01.3f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            parcial_loss = 0
            start_time = time.time()
            
    return total_loss / n_batches

def validation(model):
    model.eval()
    total_loss = 0.
    
    with torch.no_grad():
        n_batches = val_data.shape[0] // args.batch_size
        aux_val = list(range(val_data.shape[0]))
        for n in range(n_batches):
            batch, x, y = val_dataset[aux_val]
            aux_val = [x for x in aux_val if x not in batch]
            
            output = model(x)
            loss = criterion(output.view(-1), y).item()
            total_loss += loss
    return total_loss / n_batches

epochs = args.epochs
best_val_loss = float('inf')

aux_lr = str(args.lr).split('.')
model_path = aux_lr[-1]+'_'+str(args.batch_size)+'_'+str(args.epochs)
best_model_params_path = os.path.join(args.load_model, model_path+'.pt')

print('Starting training...')
count = 0
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    
    total_train_loss = train(model)
    with open('./losses/train_'+model_path+'.txt', 'a') as f:
        f.write(str(total_train_loss)+',')
        f.close()
    
    total_val_loss = validation(model)
    with open('./losses/val_'+model_path+'.txt', 'a') as f:
        f.write(str(total_val_loss)+',')
        f.close()
    val_ppl = math.exp(total_val_loss)
    
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {total_val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if total_val_loss < best_val_loss:
        count = 0
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), best_model_params_path)
    elif count > args.patience:
        print('Early stopping...')
        break
    else:
        count += 1

model.load_state_dict(torch.load(best_model_params_path))

print('Getting annotations...')
annotations = pd.read_json('./MLChallenge_Dataset/annotations.json')
img_id = list(annotations.columns)
for i in range(len(img_id)):
    aux = img_id[i].split('/')
    img_id[i] = aux[1]+'_'+aux[-1][:-4]

print('Testing...')
def test(model):
    model.eval()
    
    test_dict = {
        'img': [],
        'predicted': [],
        'target': [],
        'spoof_type': [],
        'illumination': [],
        'environment':[]
    }
    
    with torch.no_grad():
        aux_test = list(range(test_data.shape[0]))
        for n in range(test_data.shape[0]):
            batch, x, y = test_dataset[aux_test]
            aux_test = [x for x in aux_test if x not in batch]
            y = y.cpu()
            
            output = model(x)
            output = output.cpu()
            
            path = test_data['path'][batch[0]]
            aux_path = path.split('/')
            aux_path = aux_path[-1].split('_')
            aux_path = aux_path[0]+'_'+aux_path[1]
            idx = img_id.index(aux_path)
            
            test_dict['img'].append(path)
            test_dict['predicted'].append(round(float(output.view(-1)[0]), 3))
            test_dict['target'].append(float(y[0]))
            test_dict['spoof_type'].append(annotations[annotations.columns[idx]][40])
            test_dict['illumination'].append(annotations[annotations.columns[idx]][41])
            test_dict['environment'].append(annotations[annotations.columns[idx]][42])
    
    return test_dict

test_dict = test(model)
pd.DataFrame.from_dict(test_dict).to_csv('./predictions/'+model_path+'.csv', index = False)