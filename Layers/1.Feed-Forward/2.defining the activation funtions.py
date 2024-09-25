#in the previous code we have defined the model's layers in our Feed-Forward Neural Network.
#now we will define the activation functions for the layers.

#importing the dependencies
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
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
'''An activation funtion is defined as a funtion which affects how the weights of a neural network are updated.
   It is a non-linear function that is applied to the output of a neuron or a layer in a neural network,
   erves to introduce non-linearity into the network and allows the network to learn complex patterns in the data.
   
   >>>Note:"Activation funtion basically tells if the Neuron should activate or not based on the input we give it"
   
   there are diffrent kinds of activation funtions depending on what are your usecases,
   some of them are as follows:
   1. ReLU activation funtion
   2. Sigmoid activation funtion
   3. Softmax activation funtion
   4. Tanh activation function
   5. ELU activation funtion
   6. Leaky ReLU activation funtion'''

"""now that we know about the activation funtions, well move to see how we can apply the activation funtions to the layers,
   move to 3.applying the activation funtions.py"""