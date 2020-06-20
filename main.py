import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import math

import argparse
from models import *


def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    discriminator = Discriminator().to(device)

    ## loss 
    classifier_criterion = nn.NLLLoss()
    discriminator_criterion = nn.NLLLoss()

    ## learning rate and domain adaptation
    p = 0  # training progress changes linearly from 0 to 1
    alpha = 10
    bta = 0.75
    lmda = 0
    lr = args.lr

    ## optimizer
    params = list(feature_extractor.parameters())+list(classifier.parameters())+list(discriminator.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # train
    for epoch in range(1, args.epochs+1):

        train_length = min(len(svhn_train), len(mnist_train))
        svhn_train_iter = iter(svhn_train_loader)
        mnist_train_iter = iter(mnist_train_loader)
        
        i = 0
        while i < train_length:
            # dataset loading
            svhn_img, svhn_label = svhn_train_iter.next()
            mnist_img, _ = mnist_train_iter.next()

            svhn_img = svhn_img.to(device)
            mnist_img = mnist_img.to(device)
            svhn_label = svhn_label.to(device)

            optimizer.zero_grad()

            # Classifier training 
            yf = feature_extractor(svhn_img)
            y = classifier(yf.view([args.batch_size, -1]))
            
            C_loss = classifier_criterion(y, svhn_label)

            # Discriminator 

            # Domain adaptation regularizer from current domain 
            yf = feature_extractor(svhn_img)
            y = discriminator(yf.view([args.batch_size, -1]))

            d_label = torch.zeros(args.batch_size).long().to(device)
            src_loss = discriminator_criterion(y, d_label)

            # ... from target domain.
            yf = feature_extractor(mnist_img)
            y = discriminator(yf.view([args.batch_size, -1])) 

            d_label = torch.ones(args.batch_size).long().to(device)
            target_loss = discriminator_criterion(y, d_label)

            D_loss = lmda*(src_loss + target_loss)
            loss = C_loss + D_loss
            loss.backward()

            # update parameter
            optimizer.step()

            if (i % 100 == 0):
                print("Epoch %d: [%d itr] C_loss: %f, D_loss: %f" %(epoch, i, C_loss, D_loss))
                
            # update domain classifier
            p = (i+ epoch*train_length) / (args.epochs *train_length)
            lr = args.lr / ((1+alpha*p)**bta) # TODO 
            lmda = 2 / (1+ math.exp(-args.gamma*p)) -1
            i += 1 

            # TODO : test!

        torch.save({
        'feature' : feature_extractor.state_dict(),
        'discriminator' : discriminator.state_dict(),
        'classifier' : classifier.state_dict(),
        'optimF' : optimF.state_dict(),
        'optimD' : optimD.state_dict(),
        'paramC' : optimC.state_dict(),
        'args' : args
        }, 'drive/My Drive/checkpoint/model_epoch{}'.format(epoch))

# def test():


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.01, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--gamma", default=10, type=int)
    parser.add_argument("--theta", default=1, type=int)
    args = parser.parse_args()

    main(args)