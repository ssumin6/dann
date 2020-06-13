import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import argparse
from models import *


def main(args, ITE=0):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## dataset
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                            std=[0.5])
    ])
    # source    
    svhn_train = datasets.SVHN(root='data/', split='train', transform=transform, download=True)
    svhn_test = datasets.SVHN(root='data/', split='test', transform=transform, download=True)
    svhn_train_loader = DataLoader(dataset=svhn_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    svhn_test_loader = DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # target
    mnist_train = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
    mnist_train_loader = DataLoader(dataset=mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True)    
    mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    ## model
    feature_extractor = 
    classifier = 
    discriminator = 

    ## loss
    classifier_criterion = 
    discriminator_criterion = 

    ## optimizer
    optimizer = 

    # train
    for epoch in range(1, args.epochs+1):

        train()
        test()

def train():

def test():


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.01, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--gamma", default=10, type=int)
    parser.add_argument("--theta", default=1, type=int)
    args = parser.parse_args()

    main(args)