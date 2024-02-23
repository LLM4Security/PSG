import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm
from torch.optim import lr_scheduler, AdamW
from transformers import get_linear_schedule_with_warmup

class BertFeatureExtractor(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertFeatureExtractor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, attention_mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        return pooled_output

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 加载预训练的中文BERT模型和tokenizer
model = BertFeatureExtractor()
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 该数据读入
with open('./test_json_data.json') as f:
    page_data=json.load(f)

data = []


for key in page_data.keys():
    data.append(page_data[key])

tokenized_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt", max_length=128)

Dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
dataloader = DataLoader(Dataset, batch_size=16, shuffle=True)

for i in tqdm(dataloader):
    input_ids = i[0].to(device)
    attention_mask = i[1].to(device)
    outputs = model(input_ids, attention_mask)
    # 处理outputs
    print(outputs, outputs.shape)
