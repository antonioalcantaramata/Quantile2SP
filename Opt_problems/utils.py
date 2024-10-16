import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TabularDataset(Dataset):
    def __init__(self, X, y):
        """
        Characterizes a Dataset for PyTorch
        """
        
        self.n = X.shape[0]
        
        self.y = y.astype(np.float32).values.reshape(-1, 1)
        self.X = X.astype(np.float32).values
    
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.X[idx]]



class QuantFCNN(nn.Module):
    def __init__(self, input_size, quantiles, hidden_layers, hidden_size, drop=0.05):
        super().__init__()
        self.input_size = input_size
        self.output_size = len(quantiles)
        self.quantiles = quantiles
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.drop = drop
        
        # First Layer
        first_lin_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # Hidden Layers
        self.layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(self.hidden_layers - 1)
            ]
            + [nn.Linear(self.hidden_size, self.output_size)]
        )
        
        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(self.drop) for l in self.layers]
        )
        
    def forward(self, x):
        
        for i, (lin, d) in enumerate(zip(self.layers, self.droput_layers)):
            if i != self.hidden_layers:
                x = F.relu(lin(x))
                x = d(x)
            else:
                x = lin(x)
        
        return x


class FeedforwardIncremental(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dropout_rate=0.05):
        super(FeedforwardIncremental, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Output layer with ReLU for all neurons except the first one
        output_layer = nn.Linear(hidden_size, output_size)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        hidden_output = self.model[:-1](x)
        
        # Apply ReLU for all neurons in the output layer except the first one
        output = self.model[-1](hidden_output)
        output[:, 1:] = torch.relu(output[:, 1:])
        
        # Perform cumulative sum for incremental behavior
        incremental_output = torch.cumsum(output, dim=1)

        return incremental_output




class QuantileLoss(nn.Module):
    def __init__(self, quantiles, l1_pen=0, l2_pen=0):
        super().__init__()
        self.quantiles = quantiles
        self.qv = torch.tensor(quantiles, dtype=torch.float).unsqueeze(0).to(device)
        self.qv_1 = self.qv - 1
        self.l1_pen = l1_pen
        self.l2_pen = l2_pen

    def forward(self, model, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        errors = target - preds
        losses = torch.max((self.qv_1) * errors, self.qv * errors).mean(axis=0)
        loss = losses.mean()
        
        l1 = 0
        l2 = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1 += torch.norm(param, 1)
                l2 += torch.pow(torch.norm(param, 2), 2)
        
        loss += self.l1_pen*l1
        loss += self.l2_pen*l2
        
        return loss




class QNN_obj(object):
    
    def __init__(self, X_train, X_val, y_train, y_val, incremental, n_hidden, n_nodes, drop, iters, learning_rate, batch_s, l1_reg, l2_reg, opti, quantiles):
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.incremental = incremental
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes
        self.drop = drop
        self.iters = iters
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_size = batch_s
        self.opti = opti
        self.quantiles = quantiles
    
    def train(self):
        
        train_ds = TabularDataset(self.X_train, self.y_train)
        val_ds = TabularDataset(self.X_val, self.y_val)
        train_dl = DataLoader(train_ds, batch_size = self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=len(val_ds))
        
        seed = 0
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        
        criterion = QuantileLoss(self.quantiles, l1_pen=self.l1_reg, l2_pen=self.l2_reg)
        if self.incremental == True:
            model = FeedforwardIncremental(input_size=self.X_train.shape[1], hidden_size=self.n_nodes, output_size=len(self.quantiles), num_hidden_layers=self.n_hidden, dropout_rate=self.drop).to(device)
        else:
            model = QuantFCNN(input_size=self.X_train.shape[1], quantiles=self.quantiles, hidden_layers=self.n_hidden, hidden_size=self.n_nodes, drop=self.drop).to(device)
        
        if self.opti == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)
        elif self.opti == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr = self.learning_rate)
        elif self.opti == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr = self.learning_rate)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        model.apply(init_weights)
        
        print(model)
        print(device)
        print('Training')
        
        best_loss = 10000
        for epoch in range(self.iters):
            ###### Training ######
            model = model.train()
            for y, X in train_dl:
                optimizer.zero_grad()
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(model, pred, y)
                loss.backward()
                optimizer.step()
            
            ###### Validation ######
            model = model.eval()
            val_loss = 0
            with torch.no_grad():
                for y_val, X_val in val_dl:
                    X_val = X_val.to(device)
                    y_val  = y_val.to(device)
                    pred = model(X_val)
                    loss = criterion(model, pred, y_val)
                    val_loss += loss.item()
            val_loss /= float(len(val_dl))
            
            if val_loss < best_loss:
                torch.save(model.state_dict(), 'best_qnn_model.pt')
                best_loss = val_loss
                last_save = epoch
            
            if (epoch%200) == 0:
                print('epoch', epoch, 'loss', val_loss)
        
        model.load_state_dict(torch.load('best_qnn_model.pt'))
                
        print('(I)QNN fitting process completed with a validation Quantile loss of', best_loss, 'in epoch', last_save)
        return best_loss, model

