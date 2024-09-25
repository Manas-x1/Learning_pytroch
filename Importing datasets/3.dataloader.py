#this code will explain the dataloader
#the dataloader is bascially used to load the dataset into the model that we define 
#it loads the data into batches of a specified size and also shuffles the data

#dependencies
from torchvision import datasets , transforms
from torch.utils.data import DataLoader #this calls the dataloader module

#defining transformations
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

#importing datasets
x_train = datasets.MNIST(root = './data',train = True,transform = transforms)
x_test = datasets.MNIST(root = './data',train = False,transform = transforms)

#once the transformations are applied to the dataset, we can load the dataset into the dataloader

#defining the dataloader
train_load = DataLoader(x_train,batch_size = 64,shuffle = True)
test_load = DataLoader(x_test,batch_size = 64,shuffle = True)

#dataset gives the specified dataset
#batch_size is the number of samples in each batch
#shuffle is a boolean value that determines whether to shuffle the data or not

'''now our data is ready to be loaded into the model'''

"""Next move to layers to learn about basic model layers like,
    Feed-Forward Neural Network and Convolutional Neural Network,
    
    Move to layers/1.feedforward.py"""