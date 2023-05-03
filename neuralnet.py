import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        ######################
        #   YOUR CODE HERE   #

        # self.fc1 = nn.Linear(32, 256)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(256, 256)
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(256, 256)
        # self.dropout3 = nn.Dropout(p=0.5)
        # self.fc4 = nn.Linear(256, 1)

        self.lstm = nn.LSTM(input_size = 36,
                            hidden_size =128,
                            num_layers =1,
                            batch_first=True)

        self.linear = nn.Linear(128, 1)

        ######################

    def forward(self, X):
        ######################
        #   YOUR CODE HERE   #
        X, _ = self.lstm(X)
        X = X[:, -1, :]
        X = self.linear(X)
        X = nn.sigmoid(X)
        return X
        ######################
