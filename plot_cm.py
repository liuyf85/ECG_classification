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

def prosess_data():
    df = pd.read_csv('train.csv')
    def parse_heartbeat_signal(signal):
        return np.array([float(x) for x in signal.split(',')])
    df['heartbeat_signals'] = df['heartbeat_signals'].apply(parse_heartbeat_signal)
    signals_tensor = torch.Tensor(np.stack(df['heartbeat_signals'].values)).unsqueeze(dim = -1)
    labels_tensor = torch.Tensor(df['label'].values)
    signals_tensor = signals_tensor.to(torch.float)
    labels_tensor = labels_tensor.to(torch.long)
    return signals_tensor, labels_tensor

class BERT(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, n_heads, n_layers, n_inter):
        super(BERT, self).__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            intermediate_size=hidden_size * n_inter,
            max_position_embeddings=300,
            vocab_size=300
        )
        self.bert = BertModel(config)
        self.cls = nn.Parameter(torch.randn(hidden_size))
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=int(hidden_size*1/3), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=int(hidden_size*1/3), out_channels=int(hidden_size*2/3), kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=int(hidden_size*2/3), out_channels=hidden_size, kernel_size=7, padding=3)
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
        x = self.conv_activation(self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        tmp = self.cls.unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, tmp), dim=1)
        outputs = self.bert(inputs_embeds=x, output_attentions = False)
        cls_output = outputs.last_hidden_state[:, -1, :]
        logits_cls = self.classifier(cls_output)
        return logits_cls


# ======== basic setting ========
input_size = 1
hidden_size = 64
num_layers = 6
num_classes = 4
num_heads = 4
num_inter = 4

num_epochs = 100
batch_size = 512
lr = 1e-3
wd = 1e-4
# ======== basic setting ========

input, label = prosess_data()
dataset = TensorDataset(input, label)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
model.load_state_dict(torch.load('bert_weighted.pth'))

model.eval()
correct = 0
total = 0
confusion_matrix = np.zeros((4, 4))
with torch.no_grad():
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim = 1)
        
        # from time import time
        # for i in range(len(predicted)):
        #     if predicted[i] != labels[i]:
        #         print(f'distribution: {outputs.data[i]}, predicted: {predicted[i]}, labels: {labels[i]}')
                # time.sleep(0.1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            confusion_matrix[true_label, pred_label] += 1
accuracy = correct / total
# print(f"Test Accuracy: {accuracy:.2%}")
print(f'Accuracy: {accuracy:.2%}')

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Reds)
fig.colorbar(cax)
ax.set_xlabel('Predicted Labels', fontsize=14)
ax.set_ylabel('True Labels', fontsize=14)
ax.set_xticks(range(confusion_matrix.shape[1]))
ax.set_yticks(range(confusion_matrix.shape[0]))
ax.set_xticklabels(range(confusion_matrix.shape[1]), fontsize=12)
ax.set_yticklabels(range(confusion_matrix.shape[0]), fontsize=12)
for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=12)
output_path = f"confusion_matrix_test.png"
plt.title("Confusion Matrix", fontsize=16)
plt.savefig(output_path)
plt.close()