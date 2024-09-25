#in this code we will define the Convolutional Neural Network.(CNN)

#importing the dependencies\
import torch
import torch.nn as nn #this is the nn module.

#defining the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #defining the Convolutional layers.
        self.conv1 = nn.Conv2d(1,16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1)
        '''(1,16),(16,32) are the input and output channels respectively.
           kernel_size=3, stride=1, padding=1 are the parameters of the convolution layer.

           where,
           kernel_size=3 means that the kernel size of the convolution layer is 3x3.
           stride=1 means that the pixel step of the convolution layer is 1.
           padding=1 means that the padding of the convolution layer around the edges is 1.'''
        
        #defining the pooling layers.
        '''Maxpooling basically reduces the output size  of the convolutional layers by the factor of 2.
           i.e, 28*28 -> 14*14 -> 7*7
           It is used to reduce the number of parameters in the model and to prevent overfitting.'''
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           #28*28 -> 14*14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
           #14*14 -> 7*7
        
        #defining the fully connected layers.
        self.fc1 = nn.Linear(32*7*7,128)
        self.fc2 = nn.Linear(128,64)
        self.fc = nn.Linear(64,10)
        '''(32*7*7,128) is the input and output size of the fully connected layer.
            where,  "32" is the output of convolutional layer.
                    "7*7" is the reduced output from the pooling.'''
        
        #defining the activation functions.
        self.relu = nn.ReLU()
        '''this defines our activation funtion as ReLU.'''


        def forward(self,x):
        #this applies the activation funtion to the output of the convolutional layers.
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
        
        #this applies the pooling to the output of the convolutional
            x = self.pool1(x)
            x = self.pool2(x)
        #this flattens the output of the convolutional layers.
            x = x.view(-1,32*7*7)
        
        #this applies the activation funtion to the output of the fully connected layers.
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc(x)
        #this returns the output of the model, its activation will be handeled by the loss funtion.

        return x

'''this basically concludes our CNN model.
    now that we have both Feed Forward Neural Network and Convolutional Neural Network.
    we will now move to the model's attribute. to learn about model initialization and 
    its diffrent attributes.'''

"""next up move to the model's attribute"""
