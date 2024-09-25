'''once the model is defined, we need to initialize it.
for example, we have this basic feed-forward neural network.'''

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

""" now to initialize the model, we need to create an instance of the model.
    by simply calling it as a class object."""

model = Net()

"""now that our model is initialized, we can start training it,
    but we will need some loss function and optimizer to train the model."""

""" refer to next file to learn more about the loss function and optimizer."""