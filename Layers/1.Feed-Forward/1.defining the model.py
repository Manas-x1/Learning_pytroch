#in this code file we will basically define the model.
#itll be a simple linear Feed-Forward Neural Network.

#importing the dependencies
import torch
import torch.nn as nn #this is the neural network module.

#defining the model
'''model is defined as a class , which inherits from nn.Module class.
   the class has a constructor that initializes the model with the specified input size and hidden size.
   the forward method defines the forward pass of the model, where the input is passed through the linear layers and the output is returned.
   For example, the model is instantiated with input size of 784 and hidden size of 128 and will return the output of the model in 10 objects.'''

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #this is the Input layer of the model.
        self.l1 = nn.Linear(784,512) 
        #these are the Hidden layers of tht model.
        self.l2 = nn.Linear(512,256) 
        self.l3 = nn.Linear(256,128)
        self.l4 = nn.Linear(128,64)
        self.l5 = nn.Linear(64,32)
        self.l6 = nn.Linear(32,16)
        #this is the Output layer of the model.
        self.ln = nn.Linear(16,10)


"""this is the model that we will be using for the MNIST dataset,
   it is the most basic model that we can use for the image recognition task."""

"""Next up well learn about the activation funtions,
   move to 2.defining the activation functions.py"""