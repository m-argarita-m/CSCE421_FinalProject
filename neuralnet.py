import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        ######################
        #   YOUR CODE HERE   #

        self.fc1 = nn.Linear(32, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 256)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(256, 1)

        ######################

    def forward(self, X):
        ######################
        #   YOUR CODE HERE   #

        X = X.view(-1, 32)
        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)
        X = F.relu(self.fc3(X))
        X = self.dropout3(X)
        X = self.fc4(X)
        X = F.sigmoid(X)
        return X
        ######################
