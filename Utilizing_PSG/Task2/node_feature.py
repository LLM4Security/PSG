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
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# Word Embedding + Attention

glove = GloVe(name='6B', dim=50)

def feature_extract(data):

    features = []
    #Word embedding
    for key in data:
        descript = data[key] 
        # key feature
        key_ft = glove.get_vecs_by_tokens(key.split('.'))
        descript_ft = glove.get_vecs_by_tokens(descript.split())

        feat = np.concatenate((key_ft, descript_ft), axis = 0)
        features.append(feat)
    

    features = np.vstack(features)
    # key和description的拼接
    # feature = np.concatenate((keys, descripts), axis=1)
    feature = torch.from_numpy(features).float()
    # print(feature.shape)
    return feature
    

# 点数据处理
def prepare_data():
    node_path = "DeUEDroid_page"
    classes = os.listdir(node_path)

    feature_path = 'Node_feature'
    labels = []
    features = []
    for i, cls in enumerate(classes):
        
        if not os.path.exists(os.path.join(feature_path, cls)):
            os.makedirs(os.path.join(feature_path, cls))
        node_files = os.listdir(os.path.join(node_path, cls))
        for node_file in tqdm(node_files):
            try:
                with open(os.path.join(node_path, cls, node_file, 'page.json'), 'r') as f:
                    page_data = json.load(f)
                    feature = feature_extract(page_data) # 是一个长度
                    # print(feature.shape)
                    features.append(feature)
                    labels.append(i)
            except:
                # print(node_file)
                print(os.path.join(node_path, cls, node_file))
                print(sys.exc_info())
            # break
    return features, labels
    # torch.save(features, os.path.join(feature_path, cls, 'features.pt'))

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class TextAttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(TextAttentionModel, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 注意力层
        self.attention = SelfAttention(hidden_dim)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM输入： x的形状 (batch_size, sequence_length, embedding_dim)
        # LSTM输出： output的形状 (batch_size, sequence_length, hidden_dim)
        # 注意力层输出： attended的形状 (batch_size, hidden_dim)

        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out)

        # 全连接层
        out = self.fc(attended)

        return out

def custom_pad(batch):
    datas, labels = zip(*batch) 
    lengths = [len(data[0]) for data in datas]
    padded_datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return padded_datas, labels

def train(features, labels, models, epoches):
    # zip_list = list(zip(features, labels))
    # random.shuffle(zip_list)
    # features, labels = zip(*zip_list)
    dataset = DataLoader(list(zip(features, labels)), collate_fn = custom_pad, batch_size=8, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)    
    
    for epoch in range(epoches):
        model.train()
        avg_loss = 0
        for data, label in tqdm(dataset):
            # data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            data = data.to(device)
            output = model(data)
            
            # print(f'data {type(data)} label {type(label)}')
            loss = criterion(output, torch.tensor(label).to(device))
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = avg_loss / len(dataset)
        print(f'Epoch: {epoch}, Loss: {avg_loss}')
        evaluate(models, dataset, criterion)

def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            output = model(data)

            loss = criterion(output, torch.tensor(label).to(device))
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == torch.tensor(label).to(device)).sum().item()
            total_samples += len(label)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')
    return average_loss, accuracy


if __name__ == "__main__":

    features, labels = prepare_data()
    
    embedding_dim = 50  # 假设使用GloVe模型的维度
    hidden_dim = 64
    output_dim = 4  #分类结果
    EPOCHES = 1000 

    # device = torch.device('cuda:0')
    # 创建模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = TextAttentionModel(embedding_dim, hidden_dim, output_dim).to(device)
    # 训练
    train(features, labels, model, EPOCHES)
    torch.save(model.state_dict(), 'model_1000.pt')

    # output = model(embedded_text.unsqueeze(0))
    # print("Model Output Shape:", output.shape)
    
    