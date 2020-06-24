import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

import numpy as np

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
        
        start_steps = (epoch-1)*len(svhn_train_loader)
        total_steps = args.epochs *len(svhn_train_loader)
        
        for batch_idx, (sdata, tdata) in enumerate(zip(svhn_train_loader, mnist_train_loader)):
            # update domain classifier
            p = (batch_idx+ start_steps) / total_steps
            lr = args.lr / ((1+alpha*p)**bta)
            lmda = 2. / (1. + np.exp(-args.gamma*p)) -1

            # dataset loading
            svhn_img, svhn_label = sdata
            mnist_img, _ = tdata

            svhn_img = svhn_img.to(device)
            mnist_img = mnist_img.to(device)
            svhn_label = svhn_label.to(device)

            optimizer.zero_grad()

            # Classifier training 
            svhn_feature = feature_extractor(svhn_img)
            svhn_pred = classifier(svhn_feature)
            C_loss = classifier_criterion(svhn_pred, svhn_label)

            # Discriminator
            # Domain adaptation regularizer from current domain 
            svhn_feature2 = feature_extractor(svhn_img)
            src_pred = discriminator(svhn_feature2, lmda)

            d_label = Variable(torch.zeros(args.batch_size).long().to(device))
            
            src_loss = discriminator_criterion(src_pred, d_label)

            # ... from target domain.
            tgt_feature = feature_extractor(mnist_img)
            tgt_pred = discriminator(tgt_feature, lmda) 

            d_label2 = Variable(torch.ones(args.batch_size).long().to(device))
            tgt_loss = discriminator_criterion(tgt_pred, d_label2)

            D_loss = src_loss + tgt_loss
            loss = C_loss + D_loss
            loss.backward()

            # update parameter
            optimizer.step()

            if ((batch_idx+start_steps) % 100 == 0):
                print("Epoch %d: [%d itr] C_loss: %f, D_loss: %f" %(epoch, batch_idx+start_steps, C_loss.item(), D_loss.item()))

        test(feature_extractor, classifier, svhn_test_loader, device, "svhn")
        test(feature_extractor, classifier, mnist_test_loader, device, "mnist")

        if (epoch % 25 == 0):
            torch.save({
            'feature' : feature_extractor.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'classifier' : classifier.state_dict(),
            'optim' : optimizer.state_dict(),
            'args' : args
            }, 'drive/My Drive/checkpoint/model_epoch{}'.format(epoch))

def test(feature, model, data_loader, device, name):
    feature.eval()
    model.eval()
    n_total = 0 
    n_correct = 0 
    
    test_iter = iter(data_loader)
    num = len(data_loader)

    i = 0 
    while i < num:
        img, label = test_iter.next()
        # Get batch size
        batch_size = len(label)
        # Transfer data tensor to GPU/CPU (device)
        img = img.to(device)
        label = label.to(device)

        ft = feature(img)
        pred = model(ft.view(batch_size, -1))
        pred = pred.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        i += 1

    acc = n_correct.data.numpy() * 1.0 / n_total

    print("##########")
    print("TEST data : %s accuracy : %f" %(name, acc))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.01, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--gamma", default=10, type=int)
    parser.add_argument("--theta", default=1, type=int)
    args = parser.parse_args()

    main(args)