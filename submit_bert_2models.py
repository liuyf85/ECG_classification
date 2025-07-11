import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
output_file = "submit_five_vote.csv"

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

data = pd.read_csv(input_file)

def predict_labels(heartbeat_signals):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_test = BERT(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
    model_test.load_state_dict(torch.load('bert_testp_testparam.pth'))
    
    model_weighted = BERT(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
    model_weighted.load_state_dict(torch.load('bert_weighted.pth'))
    
    model_test2 = BERT(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
    model_test2.load_state_dict(torch.load('bert_testp_testparam_2.pth'))
    
    model_weighted2 = BERT(input_size, hidden_size, num_classes, num_heads, num_layers, num_inter).to(device)
    model_weighted2.load_state_dict(torch.load('bert_weighted_2.pth'))
    
    model_twoclass = BERT(input_size, hidden_size, 2, num_heads, num_layers, num_inter).to(device)
    model_twoclass.load_state_dict(torch.load('bert_two_class.pth'))
    
    test_data = list(map(float, heartbeat_signals.split(',')))
    test_data = torch.Tensor(test_data).to(device).unsqueeze(0).unsqueeze(-1)
    
    model_test.eval()
    model_weighted.eval()
    model_test2.eval()
    model_weighted2.eval()
    model_twoclass.eval()
    with torch.no_grad():
        outputs_test = model_test(test_data)
        outputs_weighted = model_weighted(test_data)
        outputs_test2 = model_test2(test_data)
        outputs_weighted2 = model_weighted2(test_data)
        outputs_twoclass = model_twoclass(test_data)
        
        v_test = outputs_test.data
        v_weighted = outputs_weighted.data
        v_test2 = outputs_test2.data
        v_weighted2 = outputs_weighted2.data
        v_twoclass = outputs_twoclass.data
        v_total = v_test + v_weighted + v_test2 + v_weighted2
        v_total[:, 0] = v_total[:, 0] + v_twoclass[:, 0]
        v_total[:, 1:] = v_total[:, 1:] + v_twoclass[:, 1:]
        _, predicted = torch.max(v_total, dim = 1)
        
        _, predicted_test = torch.max(outputs_test.data, dim = 1)
        _, predicted_weighted = torch.max(outputs_weighted.data, dim = 1)
        _, predicted_test2 = torch.max(outputs_test2.data, dim = 1)
        _, predicted_weighted2 = torch.max(outputs_weighted2.data, dim = 1)
        _, predicted_twoclass = torch.max(outputs_twoclass.data, dim = 1)
        vote = torch.zeros(4).to(predicted_test.device)
        vote[predicted_test] += 1
        vote[predicted_weighted] += 1
        vote[predicted_test2] += 1
        vote[predicted_weighted2] += 1
        if predicted_twoclass == 0:
            vote[0] += 1
        else:
            vote[1:] += 1
        max_vote, predicted_vote = torch.max(vote, dim = -1)
        for idx in range(4):
            if idx != predicted_vote and vote[idx] == max_vote:
                predicted_vote = predicted
                break
        
        # v_test_softmax = F.softmax(v_test, dim=-1)
        # v_weighted_softmax = F.softmax(v_weighted, dim=-1)
        # _, predicted2 = torch.max(v_test_softmax + v_weighted_softmax, dim = 1)
            
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
