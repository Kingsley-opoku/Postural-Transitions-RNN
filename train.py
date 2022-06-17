from argon2 import Parameters
from inflection import parameterize
import numpy as np
import torch as T
from torch import nn
import torch.nn.functional as F
from model import RNN

def train_model(model, epochs, lr):
    params = model.parameters()
    optimizer = T.Adam(params, lr)
    for i in range(epochs):
        model.train()

        train_x, train_y = next_batch(batch_size, n_steps, df, 9)

        train_loss = 0.0
        total = 0
        
        y_pred = model(train_x)
        optimizer.zero_grad()
        loss = F.cross_entropy(y_pred, train_y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()*train_y.shape[0]
        total += train_y.shape[0]

        if i % 5 == 1:
            print(f'Epoch: {i} | Loss: {train_loss}')