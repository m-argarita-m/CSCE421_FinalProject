import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        ######################
        #   YOUR CODE HERE   #

        self.fc1 = nn.Linear(30, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

        ######################

    def forward(self, X):
        ######################
        #   YOUR CODE HERE   #

        X = X.view(-1, 30)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        X = F.sigmoid(X)
        return X
        ######################
