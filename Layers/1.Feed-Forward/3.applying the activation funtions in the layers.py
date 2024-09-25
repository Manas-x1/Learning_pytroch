#in the previous code we have defined the activation funtions for the layers.
#now we will apply the activation funtions in the layers.

import torch
import torch.nn as nn #this is the neural network module.

#defining the model

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
        #defining the activation functions for the layers.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

'''in order to apply the activation funtions in the layers we have to define the forward mathod,
   the forward method defines the forward pass of the model, where the input is passed through the linear layers and the output is returned.
   the activation funtions are applied to the output of the linear layers.'''

def forward(self,x):
        x = self.relu(self.l1(x)) 
        x = self.relu(self.l2(x)) 
        x = self.sigmoid(self.l3(x)) 
        x = self.relu(self.l4(x)) 
        x = self.sigmoid(self.l5(x)) 
        x = self.relu(self.l6(x))
        #activation funtions are applied to the output of all the linear layers. till the last linear layer.
        x = self.ln(x)
        # the output of the last linear layer is returned. and it will be handeled by the loss funtion.
        return x

'''Our Feed-Forward model is now ready to be trained and tested.
   but first we need to learn about the Convolutional Neural Network.
   so move to 2.Convolutional Neural Network.'''