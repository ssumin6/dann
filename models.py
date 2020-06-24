import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, lmda):
        ctx.lmda = lmda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() *ctx.lmda

        return output, None

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        
    def forward(self, x):
        # train on SVHN
        batch_size = x.data.shape[0]
        # x = x.expand(batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        return x.view(batch_size, -1)

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
        x = F.log_softmax(self.fc3(x), 1)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        
    def forward(self, x, lmda):
        x = GRL.apply(x, lmda)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), 1)
        return x