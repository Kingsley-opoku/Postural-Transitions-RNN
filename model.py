import torch as T
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__ (self, input_size, hidden_size, num_layers, batch_size, seq_len):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.batch_size * self.hidden_size, 512)
        self.fc2 = nn.Linear(512, 258)
        self.fc3 = nn.Linear(258, 64)
        self.fc4 = nn.Linear(64, 12)

    def forward(self, x):
        h_0 = T.zeros((self.num_layer))
        rnn_out , h_n = self.rnn(x, h_0)
        last_hidden = h_n[-1]
        x = F.relu(last_hidden.flatten())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
