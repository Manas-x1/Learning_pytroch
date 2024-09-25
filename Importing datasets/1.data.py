#importing datasets

#basically what you use to import datasets and apply transformations to them
from torchvision import datasets , transforms

#defining transformations (basically applying the properties to the dataset according to our needs)
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

#loading the datasets(we are basically importing the datasets and applying the transformations to them)
x_train = datasets.MNIST(root = './data',train = True,download = True,transform = transforms)
x_test = datasets.MNIST(root = './data',train = False,download = True,transform = transforms)

#root is the location where the data is stored
#train = True means that we are training on the training data
#download = True means that we are downloading the data
#transform = transforms means that we are applying the transformations to the data

#also datasets has a whole bunch of datasets that we can use like MNIST,CIFAR10,CIFAR100,FashionMNIST

#verifying the downloaded dataset
print(x_train)
print(x_test)

print("""in the output we can clearly see the dataset is loaded (MNIST in this case),
         the number of samples in the training and test dataset,
         the location of the dataset,
         and the transformations applied to the dataset""")

#similarly we can also load other datasets by simply changing the dataset name

#importing FashionMNIST dataset
F_train = datasets.FashionMNIST(root = './data',train = True,download = True,transform = transforms)
F_test = datasets.FashionMNIST(root = './data',train = False,download = True,transform = transforms)

#this will download the FashionMNIST dataset and apply the transformations to it
#verifying the downloaded dataset
print(F_train)
print(F_test)

#this is basically how you can import most of the datasets for basic Neural Networks,
#and also transformations are explained into the next file

Cifar10_train = datasets.CIFAR10(root = './data',train = True,download = True,transform = transforms)
Cifar10_test = datasets.CIFAR10(root = './data',train = False,download = True,transform = transforms)
#this will download the CIFAR10 dataset and apply the transformations to it
#verifying the downloaded dataset
print(Cifar10_train)
print(Cifar10_test)

emnist_train = datasets.EMNIST(root = './data',train = True,download = True,transform = transforms,split="balanced")
emnist_test = datasets.EMNIST(root = './data',train = False,download = True,transform = transforms,split="balanced")
#this will download the EMNIST dataset and apply the transformations to it
#verifying the downloaded dataset
print(emnist_train)
print(emnist_test)

"""Next up we will be importing the datasets and applying the transformations to them,
    so move onto transformations.py""" 


        