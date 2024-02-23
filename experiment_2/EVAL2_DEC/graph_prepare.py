import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import torchtext
from torchtext.vocab import GloVe
import joblib
import sys
import json
import random
import torch.optim as optim
# from torch_geometric.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
import networkx as nx
from node_feature import SelfAttention, TextAttentionModel, glove
from torch.utils.data import Dataset, DataLoader
import traceback2
from torch_geometric.utils.convert import from_networkx



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

embedding_dim = 50  # 假设使用GloVe模型的维度
hidden_dim = 64
output_dim = 4  #分类结果
EPOCHES = 10
# node_model = torch.load('node_model.pt')
model = TextAttentionModel(embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model_1000.pt'))
model.fc = nn.Sequential() # 只获得attention layer的结果

def graph_construct(node_data, edge_data, class_id):
    nodes = {}
    edges = []
    node_features = []
    graph_nodes = []

    # print(node_data)
    # print(edge_data)
    for i, key in enumerate(node_data):
        # print(f'key = {key}')
        nodes[key] = i
        # feature获取
        key_ft = glove.get_vecs_by_tokens(key.split('.'))
        descript_ft = glove.get_vecs_by_tokens(node_data[key].split())
        feat = torch.tensor(np.concatenate((key_ft, descript_ft), axis = 0)).unsqueeze(dim = 0)
        feat = model(feat).squeeze(dim=0) #这里到底要不要squeeze

        node_features.append(feat)

    for edge in edge_data:
        if edge['scr'] in nodes and edge['dest'] in nodes:
            edges.append((nodes[edge['scr']], nodes[edge['dest']]))
        
    print('###egde length', len(edge_data))
    print('###node length', len(node_features))
    
    graph = nx.DiGraph()
    
    graph.add_nodes_from([i for i in range(len(node_features))])
    if len(edges) > 0:
        # raise Exception('No Edges!') 
        # edges = None 
        graph.add_edges_from(edges)
    else: 
        print('No Edges')
    new_node_index = len(node_features)
    graph.add_node(new_node_index)
    for i in range(len(node_features)):
        edges.append((new_node_index, i))
    graph.add_edges_from(edges)
    new_node_feature = 0 * torch.ones_like(node_features[0])  # Assuming the feature size is the same
    node_features.append(new_node_feature)
 
    # 将图的边转换为PyTorch Geometric中的边索引表示
    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()

    node_features = torch.tensor(np.array([item.cpu().detach().numpy() for item in node_features])).cuda()
    
    # 构建PyTorch Geometric的Data对象
    graph_data = Data(x=node_features, edge_index=edge_index)
    print(f'### graph info {graph_data}')
    return graph_data

def prepare_data():
    edge_path = 'DeUEDroid_result'
    node_path = 'DeUEDroid_page'
    classes = os.listdir(edge_path)
    graphs = []
    labels = []
    hashs = []
    for i, cls in enumerate(classes):
        # cls_path = os.path.join(edge_path, cls)
        # edge_file = os.listdir(os.path.join(edge_path, cls))
        valid_files = os.listdir(os.path.join(node_path, cls))
        for file in tqdm(valid_files): 
            try:
                print('#### file name ', file)
                edge_data = json.load(open(os.path.join(edge_path, cls, file, f'{file}.json'), 'r'))
                node_data = json.load(open(os.path.join(node_path, cls, file, 'page.json'), 'r'))
                graph_data = graph_construct(node_data, edge_data['transitions'], i)
                graphs.append(graph_data)
                labels.append(i)
                hashs.append(file)
            except Exception as e:
                print(f'Error processing {file}: {e}')
                print(traceback2.format_exc())
        # break
    return graphs, labels, hashs

class GraphDataset(Dataset):
    def __init__(self, graphs, labels, hashs):
        self.graphs = graphs
        self.labels = labels
        self.hashs = hashs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index], self.hashs[index]
    


def collate_fn(data):
    # print(data)
    graphs, labels, hashs = map(list, zip(*data))
    # print('### graphs', graphs)
    batched_graphs = Batch.from_data_list(graphs)
    # print('### collate_fn',batched_graphs.batch_size)
    # print(batched_graphs)
    return batched_graphs, labels, hashs


if __name__ == '__main__':
    graphs, labels, hashs = prepare_data()
    # print(type(graphs))
    dataset = DataLoader(GraphDataset(graphs, labels, hashs), batch_size=16, shuffle=True, collate_fn=collate_fn)
    for i in dataset:
        print(i.hashs)
    # for i, data, labels in enumerate(dataset):
    #     print(f"#### BATCH {i}#####")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.save(graphs, 'graphs2.pt')
    # torch.save(labels, 'labels2.pt')
    

    