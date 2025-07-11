import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertModel, BertConfig
from transformers import GPT2Config, GPT2Model

input_file = "testA.csv"
output_file = "submit.csv"

# ======== basic setting ========
input_size = 1
hidden_size = 64
num_layers = 6
num_classes = 4
num_heads = 4
num_inter = 2

num_epochs = 10
batch_size = 64
lr = 1e-3
wd = 1e-4
# ======== basic setting ========

class GPT2(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, n_heads, n_layers, n_inter):
        super(GPT2, self).__init__()
        config = GPT2Config(
            n_embd=hidden_size,
            n_head=n_heads,
            n_layer=n_layers,
            n_inner=hidden_size * n_inter,
            n_positions=300,
            vocab_size=300
        )
        self.gpt2 = GPT2Model(config)
        self.cls = nn.Parameter(torch.randn(hidden_size))
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=5, padding=2)
        self.conv_activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        is_training = self.training
        if is_training:
            noise = torch.normal(mean=0.0, std=0.01, size=x.size()).to(x.device)
            x = x + noise
        x = x.permute(0, 2, 1)
        x = self.conv_activation(self.conv1(x))
        x = self.conv_activation(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        tmp = self.cls.unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, tmp), dim=1)
        outputs = self.gpt2(inputs_embeds=x, output_attentions = False)
        cls_output = outputs.last_hidden_state[:, -1, :]
        logits_cls = self.classifier(cls_output)
        return logits_cls

data = pd.read_csv(input_file)

def predict_labels(heartbeat_signals):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
    model.load_state_dict(torch.load('ecg_gpt2_model.pth'))
    test_data = list(map(float, heartbeat_signals.split(',')))
    test_data = torch.Tensor(test_data).to(device).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, dim = 1)
        labels = torch.zeros(4)
        labels[predicted] = 1
        
    return labels.squeeze().cpu().numpy()


counter = 0
predictions = []
for _, row in data.iterrows():
    labels = predict_labels(row['heartbeat_signals'])
    predictions.append(labels)
    counter += 1
    print(counter)
    # if counter > 100:
    #     break

output = pd.DataFrame(predictions, columns=['label_0', 'label_1', 'label_2', 'label_3'])
output.insert(0, 'id', data['id'])

output.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
