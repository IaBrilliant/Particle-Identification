#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:47:48 2020

@author: brilliant
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from IPython.display import clear_output
from IPython.display import Image
from PIL import Image



#Transforming a set of images into a set of tensors with variables (height, width, RGB).
data_transform = transforms.Compose([transforms.ToTensor(), \
                 transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

dataset_3 = datasets.ImageFolder(root = 'Train', transform = data_transform) 
test_3 = datasets.ImageFolder(root = 'Test', transform = data_transform) 

train_loader = DataLoader(dataset_3, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_3, batch_size = 32, shuffle = True)

#Kaon - 0; Pion - 1;

class ConvClassifier1(nn.Module): 
    """Initiate the NN with 2 convolutional layers"""
    
    def __init__(self, image_size): # Upload the image
        super(ConvClassifier1, self).__init__()
        self.conv_layers1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_layers2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.linear_layers = nn.Sequential(nn.Linear(image_size//2*image_size//2*16, 2), \
                                           nn.LogSoftmax(dim=1))
    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        x = x.view(x.size(0), -1) 
        x = self.linear_layers(x)
        return x
    
def train(network, epochs, learning_rate):
    """Training the NN"""
    
    loss = nn.NLLLoss() #Initialises the loss parameter, defined as a Negative Log Likelihood function 
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate) #In this NN Adam is used as an optimiser 
    train_loss_epochs = []
    test_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []
    
    try:
        for epoch in range(epochs): 
            losses = []
            accuracies = []
            for X, y in train_loader:
                network.zero_grad() #Sets gradients calculated before to zero to allow back-propagation
                prediction = network(X) #Array of images is uploaded; the NN returns its prediction on wheights 
                loss_batch = loss(prediction, y) #Returning a loss on the batch based on predictions and true labels
                losses.append(loss_batch.item()) #Adding loss_batch to an array, stating that no grad should be calculated for this loss_batch
                loss_batch.backward() #Process of back-propagation. Calculation of anti-gradients for each weight
                optimizer.step() #Updating the weights
                accuracies.append((np.argmax(prediction.data.numpy(), 1)==y.data.numpy()).mean()) #Evaluating the accuracy of outputs
            train_loss_epochs.append(np.mean(losses)) #Taking the average for better visual representation
            train_accuracy_epochs.append(np.mean(accuracies))
            
            #Testing section:
            losses = []
            accuracies = []    
            for X, y in test_loader:
                prediction = network(X)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.item())
                accuracies.append((np.argmax(prediction.data.numpy(), 1)==y.data.numpy()).mean())
            test_loss_epochs.append(np.mean(losses))
            test_accuracy_epochs.append(np.mean(accuracies))
            clear_output(True)
            print('\rEpoch {0}... (Train/Test) NLL: {1:.3f}/{2:.3f}\tAccuracy: {3:.3f}/{4:.3f}'.format(
                        epoch, train_loss_epochs[-1], test_loss_epochs[-1],
                        train_accuracy_epochs[-1], test_accuracy_epochs[-1]))
            
            #Plotting graphs, depicting loss and accuracy of training and test processes against epochs. 
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_epochs, label='Train loss')
            plt.plot(test_loss_epochs, label='Test loss')
            plt.xlabel('Epochs', fontsize=13)
            plt.ylabel('Loss', fontsize=13)
            plt.legend(loc=0, fontsize=13)
            plt.grid()
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracy_epochs, label='Train accuracy')
            plt.plot(test_accuracy_epochs, label='Test accuracy')
            plt.xlabel('Epochs', fontsize=13)
            plt.ylabel('Accuracy', fontsize=13)
            plt.legend(loc=0, fontsize=13)
            plt.grid()
            plt.savefig('Output.png', dpi = 250)
            plt.show()
            
    except KeyboardInterrupt: #Stopping the learning process with your progress saved. 
        pass

neural_network = ConvClassifier1(332)
train(neural_network, epochs=5, learning_rate=0.001)


def errors(nn): 
    for X, y in test_loader:
        predictions = nn(X)
        predictions = predictions.detach().numpy()
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions, axis=-1).reshape(-1, 1)

        prob = predictions[np.arange(y.shape[0]), y]

        ind = np.argsort(prob)[:25]

        plt.figure(figsize=(6, 7))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            Xn = X[ind[i]]
            Xn = Xn.transpose(0, 1).transpose(1, 2).numpy()
            Xn = Xn.astype('float64')
            Xn = (Xn - np.min(Xn, axis=0)) / (np.max(Xn, axis=0) - np.min(Xn, axis=0))
            Xn = np.rint(Xn * 255).astype('uint8')
            plt.imshow(Xn.reshape(332, 332, 3))
            plt.title("%d(%d) - %.2f" % (np.argmax(predictions[ind[i], :]), y[ind[i]], prob[ind[i]]))
            plt.axis('off')   
        plt.savefig("Errors.png", dpi = 250)
        break
    
errors(neural_network)