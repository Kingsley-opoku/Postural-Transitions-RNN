import numpy as np
import torch as T
from torch import nn
import torch.nn.functional as F
from model import RNN
from data_loader import DataHandler
from torch import optim

model = RNN(32, 16, 10, 32, 16)
data = DataHandler('/Users/felixschekerka/Desktop/data/HAPT Data Set/Train', '/Users/felixschekerka/Desktop/data/HAPT Data Set/Test')


def train_model(model, epochs, lr):
    params = model.parameters()
    optimizer = optim.Adam(params, lr)
    criterion = nn.MSELoss()
    for i in range(epochs):
        #setting model to training mode
        model.train()

        train_loss = 0
        total = 0

        train_x, train_y = data.df_train_batch(32, 8, 562)

        #transform them into Tensors
        train_x = T.tensor(train_x)
        train_y = T.tensor(train_y)



        train_loss = 0
        total = 0

        #Reset the Gradients
        optimizer.zero_grad()

        #Get the outputs
        y_pred = model.forward(train_x)

        #compute the loss
        train_loss = criterion(y_pred, train_y)

        #compute the gradients
        train_loss.backward()

        #apply the gradients
        optimizer.step()


        sum_loss += train_loss.item()*train_y.shape[0]
        total += train_y.shape[0]

        if i % 5 == 1:
            print(f'Epoch: {i} | Loss: {train_loss}')


def classify(model, input):
    with T.no_grad():
        logits = model.forward(input)
        ps = F.softmax(logits, dim=1)
        pred = ps.argmax()
    return pred


train_model(model, 20, 0.001)