#we will make a feed forward neural network for MNIST dataset

#importing the dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#defining the transformations
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

#loading the dataset (applying the transformations)
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform)

#creating dataloaders
train_load = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_load = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

#defining the feed forward neural network
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.l1 = nn.Linear(784,516)  # input layer
        self.l2 = nn.Linear(516,256)  # hidden layer
        self.l3 = nn.Linear(256,128)  # hidden layer
        self.l4 = nn.Linear(128,64)  # hidden layer
        self.l5 = nn.Linear(64,32)  # hidden layer
        self.l6 = nn.Linear(32,16) # hidden layer
        self.ln = nn.Linear(16,10)  # output layer
        self.relu = nn.ReLU() #relu activation function
        self.sigmoid = nn.Sigmoid() # sigmoid activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.sigmoid(self.l5(x))
        x = self.softmax(self.ln(x))
        return x

#initializing the model
model = FeedForwardNN()
