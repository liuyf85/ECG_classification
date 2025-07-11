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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=5, padding=2)
        self.conv_activation = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p = 0.2)

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
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ======== basic setting ========
input_size = 1
hidden_size = 64
num_layers = 4
num_classes = 4

num_epochs = 20
batch_size = 512
lr = 1e-3
wd = 1e-4
# ======== basic setting ========

input, label = prosess_data()
dataset = TensorDataset(input, label)

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# total_size = len(dataset)
# test_size = int(total_size * 0.1)
# train_size = total_size - test_size

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{k_folds}')

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, predicted = torch.max(outputs.data, dim = 1)
            correct = (predicted == labels).sum().item() / predicted.size(0)
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, accuracy: {correct:.2f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim = 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    fold_accuracies.append(accuracy)
    # print(f"Test Accuracy: {accuracy:.2%}")
    print(f'Fold {fold + 1} Accuracy: {accuracy:.2%}')

mean_accuracy = sum(fold_accuracies) / k_folds
print(f'Mean Accuracy over {k_folds} folds: {mean_accuracy:.2%}')