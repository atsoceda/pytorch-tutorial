import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

# params
input_size = 784  # MNIST is 28x28 pixels = 784
hidden_size = 500  # use a reasonably sized hidden layer to get good performance
num_classes = 10  # MNIST has 10 classes
batch_size = 100  # number of digits per batch
learning_rate = 0.001
num_epochs = 5


# get MNIST Dataset into "MNIST" object with attribute of data in the form of torch tensors
# MNIST objects have a __len__() method, to check number of image-label pairs
training_set = dsets.MNIST(root='./data',  # directory in which the data is remotely stored
                           train=True,     # True if the training set is desired, False if the testing set is desired
                           transform=transforms.ToTensor(),  # method to convert data to Torch tensors
                           download=True)  # Download data (seems superfluous since it will not re-download if it is
                                           # already downloaded, thus should always be True.

testing_set = dsets.MNIST(root='./data',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)  # if dataset is already downloaded will not re-download, so True has no effect

print("Num training set images: ", len(training_set))
print("Num testing set images: ", len(testing_set))

"""
An iterable, or iterator object, is defined as an object that has a  __next__() method.
Below we prepare data into an iterator object called "DataLoader", for easy batching. When the DataLoader object is 
iterated over, the __next__() method is called on the DataLoader to sequentially retrieve data in the form of Torch 
Tensors from the datasets defined above.
"""
training_loader = DataLoader(dataset=training_set,
                             batch_size=batch_size,
                             shuffle=True)

testing_loader = DataLoader(dataset=testing_set,
                            batch_size=batch_size,  # no effect of batching for testing, except for memory
                            shuffle=False)  # no effect of shuffling for testing

print("Num training batches: ", len([i for i in training_loader])) # equal to len(training_set) / batch_size
print("Num testing batches: ", len([i for i in testing_loader]))  # equal to len(testing_set) / batch_size


# Define neural network (MLP) model with one hidden layer
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# instantiate network
net = Net(input_size, hidden_size, num_classes)

# Define loss function and optimzer
criterion = nn.CrossEntropyLoss()  # define criterion as an instance of a class from the neural network module
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
"""optimizer is an instance of the Adam class that has the parameters of the nn as an atribute"""


# Training the network
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        outputs = net(images)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:

            print("epoch: [%d/%d], step: [%d/%d], loss: [%.4f]"
                  % (epoch, num_epochs, i*100, len(training_set)//batch_size, loss.data[0]))

# Test the model in batches
total = 0
correct = 0
for (images, labels) in testing_loader:  #images and labels are Torch Tensors
    images = Variable(images.view(-1, 28*28))  # must be a Variable to go forward through net.
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += len(labels)
    correct += (predicted == labels).sum()  #make sure predicted and labels are both Variables
print()
# print(correct, "\n", total)
# print(correct/total)
print("Accuracy on full test set is: %f" % (correct/total))
