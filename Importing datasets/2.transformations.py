#this will explain what transformations are and how to apply them to the dataset

#dependencies
from torchvision import datasets , transforms

#we first define the transformations before applying them to the dataset,
transforms = transforms.Compose([
                                transforms.ToTensor(), #changes the image object to a tensor
                                transforms.Normalize((0.5,),(0.5,)), #noramlizes the image data so that it is centered around 0 and has a range of [-1,1]
                                transforms.Resize((28,28)),#resizes the image to 28x28
                                transforms.RandomHorizontalFlip(p=0.5),#randomly flips the image horizontally with a probability of 0.5
                                transforms.RandomRotation(degrees=(-10,10)),#randomly rotates the image by a random angle between -10 and 10 degrees    
                                transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),#randomly translates the image by a random amount between 0.1 and 0.1 in the x and y direction   
                                transforms.RandomPerspective(distortion_scale=0.5,p=0.5),#randomly applies a perspective transformation to the image with a distortion scale of 0.5 and a probability of 0.5    
                                transforms.RandomGrayscale(p=0.1),#randomly converts the image to grayscale with a probability of 0.1 
                                ]) 
#these are just some common transformations that we can apply to the dataset
#Compose is used to chain together multiple transformations

#applying transformations to the dataset
x_train = datasets.MNIST(root = './data',train = True,transform = transforms)
x_test = datasets.MNIST(root = './data',train = False,transform = transforms)

#transformations are applied to the dataset as an argument of the dataset funtion, 
#another thing to note is that the transformations are applied to the dataset before it is loaded into the dataloader

'''Next up we will make dataloader to load the transformed data into our model,
    so now move to dataloader.py'''