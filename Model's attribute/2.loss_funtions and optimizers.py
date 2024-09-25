#now we will learn about the loss funtion and optimizer.
#but first we will import them into the code 

#importing the loss funtion and optimizer.
from torch.optim import Adam,SGD  #this will import the Adam and SGD optimizer.
import torch.nn as nn #this will let us import the loss funtion.

'''loss funtions and optimizers are defined with the initialization of the model.
    so we will define them in the model.py file.'''

model = Net()
loss_fun = nn.CrossEntropyLoss() #this will set the loss funtion as the cross entropy loss funtion.
optimizer = Adam(model.parameters(),lr=0.001) #this will set the optimizer as the Adam optimizer.


"""now to answer the question, why do we need to define the loss funtion and optimizer in the model.
    the answer is simple, because we need to use them in the training loop to refine our results with each epochs(runs).
    """

"""Loss funtions are used to calculate the loss of the model.
   which helps with the optimization of the model. by changing the weights of the layers in the next epoch

Note:"There are specific type of funtion which determine the loss of the model."
        these are as follows:
            1. Cross Entropy Loss
            2. Mean Squared Error Loss
            3. Binary Cross Entropy Loss
            4. Hinge Loss

these are the most common loss funtions used in the Deep Learning.
they basically define the loss of the model and help with the backpropogation of the model."""

"""Optimizers are used to optimize the model.
   which uses the loss from the loss funtion to optimize the model by changing the weights of the layers in the next epoch.
   which is also known as the backpropogation of the model.

   Note:"There are specific type of optimizers which are used to optimize the model.
   these are as follows:
            1. Stochastic Gradient Descent (SGD)
            2. Adam
            3. RMSprop
            4. Adagrad
            5. Adadelta
            6. Adamax
            7. Nadam
            8. Ftrl
            9. Rprop
            etc.

these are the most common optimizers used in the Deep Learning.
any one can be used as per the requirement of the model."""
            