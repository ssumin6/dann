import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

    
# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=3)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=3)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        
    def forward(self, x):
        # train on SVHN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x

# Classifier
class Classifier(nn.Module): 
    #Gy
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(128*5*5, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = GRL.apply(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class GRL(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() 

        return output, None