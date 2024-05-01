import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool,JumpingKnowledge, global_max_pool, DenseGCNConv
from torch_geometric.nn.norm import layer_norm
import numpy as np
import torchtext
from torchtext.vocab import GloVe
import joblib
import sys
import json
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
import networkx as nx
from node_feature import SelfAttention, TextAttentionModel, glove
from torch.utils.data import Dataset
import traceback2
from edge_prepare import prepare_data, GraphDataset, collate_fn
import argparse
from torch_geometric.utils import scatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler

EPOCHES = 500
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 16
INPUT_SIZE = 64
NUM_CLASSES = 4

class GraphNeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, drop):
        super(GraphNeuralNetwork, self).__init__()

        self. model = Sequential('x, edge_index, batch', [
            (nn.Dropout(p=drop), 'x -> x'),
            (GCNConv(num_features, hidden_channels), 'x, edge_index -> x1'),
            # (layer_norm.LayerNorm(64), 'x1 -> x1'),
            nn.ReLU(inplace=True),
            (GCNConv(hidden_channels, hidden_channels), 'x1, edge_index -> x2'),
            nn.ReLU(inplace=True),
            (nn.Dropout(p=drop), 'x -> x'),            
            (GCNConv(hidden_channels, hidden_channels), 'x2, edge_index -> x3'),
            nn.ReLU(inplace=True),
            (GCNConv(hidden_channels, hidden_channels), 'x3, edge_index -> x3'),
            nn.ReLU(inplace=True),
            
            (lambda x1, x2, x3: [x1, x2, x3], 'x1, x2, x3 -> xs'),
            (JumpingKnowledge("cat", hidden_channels, num_layers=3), 'xs -> x'),
            (global_mean_pool, 'x, batch -> x'),
            nn.Linear(hidden_channels*3, num_classes)
            
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index, data.batch)
        #print(x.shape)
        return F.log_softmax(x, dim=1)
    
def train(model, device, train_loader, val_loader, optimizer, scheduler, test_loader):

    for epoch in range(EPOCHES):
        
        avg_loss = 0
        model.train()
        for data, target in tqdm(train_loader):
            # print('data', data.x, data.edge_index)
            # print(type(data))
            # print(target)
            # try:
            data, target = data.to(device), torch.tensor(target, dtype = int).to(device)
            # print(data)
            # print(data)
            # print(data.x)
            # print(data.x.shape)
            optimizer.zero_grad()
            output =  model(data)
            # print(output.shape)
            # output = scatter(output, data.batch, dim=0, reduce='mean')
            # print(output.mean(dim=0))
            # print('output_shape', output.shape)
            loss = F.nll_loss(output, target)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            # except Exception as e:
            #     print(traceback2.format_exc())
            #     print(data.edge_index.shape)
        scheduler.step()
        print("@val:")
        test(model, device, val_loader)
        print("@test:")
        test(model, device, test_loader)
        avg_loss = avg_loss / len(train_loader)
        print(f'Epoch: {epoch}, Loss: {avg_loss}')
        
def test(model, device, test_loader):   
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_results = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.tensor(target, dtype = int).to(device)
            try:
                output = model(data)
            except Exception as e:
                print("error:",data)
            # output = scatter(output, data.batch, dim=0,reduce='mean')
            result = output.max(1, keepdim=True)[1]
            correct += result.eq(target.view_as(result)).sum().item()
            total += len(target)
            all_targets.extend(target.cpu().numpy())
            all_results.extend(result.cpu().numpy())
    accuracy = 100. * correct / total
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(correct, total, accuracy))
    f1 = f1_score(all_targets, all_results, labels=[0, 2, 3], average='micro')
    print('F1 Score: {:.2f}%'.format(f1 * 100))

    return accuracy,f1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', type=bool, default = False)

    args = parser.parse_args()

    if args.reload:
        graphs, labels = prepare_data()
        torch.save(graphs, 'graphs2.pt')
        torch.save(labels, 'labels2.pt')
    else:    
        if os.path.exists('graphs2.pt') and os.path.exists('labels2.pt'):
            graphs = torch.load('graphs2.pt')
            labels = torch.load('labels2.pt')
            # for g in graphs:
                # print(g.edge_index.dtype)
        else:
            graphs, labels = prepare_data()
            torch.save(graphs, 'graphs2.pt')
            torch.save(labels, 'labels2.pt')
    
    lr = [0.01,0.001,0.0001]
    step = [10,50,100,500]
    gamma = [0.9,0.8,0.7,0.6,0.5]
    hidden = [64,128,256]
    drop = [0.1,0.2,0.5,0.01]
    for lr_ in lr:
        for step_ in step:
            for gamma_ in gamma:
                for hidden_ in hidden:
                    for drop_ in drop:
                        log_file = open('auto_para.txt', 'a')
                        dataset = DataLoader(GraphDataset(graphs, labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                        # 假设 graphs 和 labels 是你的数据和标签
                        # 通过 train_test_split 进行划分，test_size 参数表示测试集的比例
                        # random_state 参数用于设置随机种子，确保划分的一致性
                        graphs_train, graphs_test, labels_train, labels_test = train_test_split(graphs, labels, test_size=0.1, random_state=42)
                        graphs_train, graphs_val, labels_train, labels_val = train_test_split(graphs_train, labels_train, test_size=0.22222, random_state=42)
                        train_loader = DataLoader(GraphDataset(graphs_train, labels_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                        val_loader = DataLoader(GraphDataset(graphs_val, labels_val), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                        

                        test_loader = DataLoader(GraphDataset(graphs_test, labels_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

                        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

                        model = GraphNeuralNetwork(INPUT_SIZE, NUM_CLASSES, hidden_, drop_).to(device) 
                        # print(model)
                        optimizer = optim.Adam(model.parameters(), lr=lr_)
                        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_, gamma=gamma_)

                        train(model, device, train_loader, val_loader, optimizer, scheduler, test_loader)
                        log_file.write('##### Result on test set:\n')
                        acc,f1 = test(model, device, test_loader)
                        log_file.write("lr:"+str(lr_)+" step:"+str(step_)+" gamma:"+str(gamma_)+" hidden:"+str(hidden_)+" drop:"+str(drop_)+"\n")
                        log_file.write("acc: "+str(acc)+" f1: "+str(f1)+"\n")
                        log_file.close()