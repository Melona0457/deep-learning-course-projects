import torch

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim


transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8,

                                          shuffle=True, num_workers=2)



dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transform)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# function to show an image

def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    print(np.transpose(npimg, (1, 2, 0)).shape)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()


# get some random training images

images, labels = next(iter(dataloader_train))



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# print size of single image

print(images[1].shape)


example_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)

print(example_conv)


example_MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

print(example_MaxPool2d)


# Define a CNN model

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()



        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=1)  # 32→26

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)                           # 26→13



        # 4x4 Convolutional layer with 16 filters, strides of 1, and ReLU activation

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1) # 13→10

        # 2x2 Max pooling layer with strides of 2

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)                           # 10→5



        # nn.Linear(input, output) / input = channels * height * width = 16*5*5

        self.fc1 = nn.Linear(16 * 5 * 5, 100)   # Fully connected layer with 100 output units and ReLU

        # Fully connected layer with 80 output units and ReLU activation

        self.fc2 = nn.Linear(100, 80)

        # Fully connected layer with 10 output units

        self.fc3 = nn.Linear(80, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.maxpool1(x)



        x = F.relu(self.conv2(x))

        x = self.maxpool2(x)



        x = torch.flatten(x, 1)   # (B,16,5,5)→(B,400)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)           # logits

        return x



# Function to train the network



def train(net, dataloader_train, max_epoch, crit, optimizer, device, model_path='./cifar_net.pth'):



    for epoch in range(max_epoch):  # loop over the dataset multiple times



        running_loss = 0.0

        for i, data in enumerate(dataloader_train, 0):

            # get the inputs; data is a list of [inputs, targets]

            inputs, targets = data



            # Training on GPU

            inputs = inputs.to(device)

            targets = targets.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inputs)

            loss = crit(outputs, targets)

            loss.backward()

            optimizer.step()



            # print statistics

            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f' %

                      (epoch + 1, i + 1, running_loss / 2000))

                running_loss = 0.0



    print('Finished Training')

    torch.save(net.state_dict(), model_path)

    print('Saved Trained Model')


PATH = './cifar_net.pth'

epoch = 2



# initialize model

net = Net()



# Training on GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = net.to(device)



# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



train(net, dataloader_train, epoch, criterion, optimizer, device, PATH)


# function to calculate accuracy

def print_accuracy(net, dataloader):

    correct = 0

    total = 0



    with torch.no_grad():

        for data in dataloader:

            images, labels = data

            # Inference on GPU

            images = images.to(device)

            labels = labels.to(device)



            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    print('Accuracy of the network on the %d test images: %d %%' % (total,

        100 * correct / total))


# load trained model then test

net.load_state_dict(torch.load(PATH))

print_accuracy(net, dataloader_test)


example_model = nn.Sequential(

          nn.Conv2d(1,20,5),

          nn.ReLU(),

          nn.Conv2d(20,64,5),

          nn.ReLU()

        )

print(example_model)


example_BatchN = nn.BatchNorm2d(100)

print(example_BatchN)


'''

About parameter

in_planes : # of input channel

n1xn1 : # of output channel for first branch

n3xn3_blue : # of output channel for second branch's 1x1 conv layer

n3xn3 : # of output channel for second branch

n5xn5_blue : # of output channel for third branch's 1x1 conv layer

n5xn5 : # of output channel for third branch

pool_planes : # of output channel for fourth branch



'''

class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3_blue, n3x3, n5x5_blue, n5x5, pool_planes):

        super(Inception, self).__init__()

        ##############################################################################

        #                          IMPLEMENT YOUR CODE                               #

        ##############################################################################

        # 1x1 conv branch

        self.b1 = nn.Sequential(

            nn.Conv2d(in_planes, n1x1, kernel_size=1, bias=False),

            nn.ReLU(inplace=True)

        )



        # 1x1 conv -> 3x3 conv branch

        self.b2 = nn.Sequential(

            nn.Conv2d(in_planes, n3x3_blue, kernel_size=1, bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(n3x3_blue, n3x3, kernel_size=3, padding=1, bias=False),

            nn.ReLU(inplace=True)

        )



        # 1x1 conv -> 5x5 conv branch

        self.b3 = nn.Sequential(

            nn.Conv2d(in_planes, n5x5_blue, kernel_size=1, bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(n5x5_blue, n5x5, kernel_size=5, padding=2, bias=False),

            nn.ReLU(inplace=True)

        )



        # 3x3 pool -> 1x1 conv branch

        self.b4 = nn.Sequential(

            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(in_planes, pool_planes, kernel_size=1, bias=False),

            nn.ReLU(inplace=True)

        )

        ##############################################################################

        #                          END OF YOUR CODE                                  #

        ##############################################################################

    def forward(self, x):

        y1 = self.b1(x)

        y2 = self.b2(x)

        y3 = self.b3(x)

        y4 = self.b4(x)

        return torch.cat([y1,y2,y3,y4], 1)






# Define the residual block class

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()



        # The first convolutional layer

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)



        # The second convolutional layer

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)



        # Shortcut connection (identity mapping)

        if stride != 1 or in_channels != out_channels:

            ##############################################################################

            #                        IMPLEMENT OF YOUR CODE                       #

            ##############################################################################

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(out_channels)

            )



            ##############################################################################

            #                           END OF YOUR CODE                          #

            ##############################################################################

        else:

            self.shortcut = nn.Identity()



    def forward(self, x):

        # Pass through the first convolutional layer

        out = self.conv1(x)

        out = self.bn1(out)

        out = nn.ReLU()(out)



        # Pass through the second convolutional layer

        out = self.conv2(out)

        out = self.bn2(out)



        # Shortcut connection

        shortcut = self.shortcut(x)



        # Add the output and the shortcut and pass it through a relu activation layer for the final output. (Residual connection implementation)

        ##############################################################################

            #                        IMPLEMENT OF YOUR CODE                       #

        ##############################################################################

        out = out + shortcut

        out = F.relu(out, inplace=True)

        ##############################################################################

            #                        IMPLEMENT OF YOUR CODE                       #

        ##############################################################################

        return out


# Define a CNN model

class BetterNet(nn.Module):

    def __init__(self):

        super(BetterNet, self).__init__()

        ##############################################################################

        #                          IMPLEMENT YOUR CODE                               #

        ##############################################################################

        # stem: 3x32x32 -> 32x32x32

        self.stem = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)

        )

        # downsample to 16x16, widen to 64

        self.res1 = ResidualBlock(32, 64, stride=2)   # 32x32 -> 64x16x16

        self.res2 = ResidualBlock(64, 64, stride=1)   # 64x16x16



        # Inception keeping channels at 64

        self.inc = Inception(

            in_planes=64,

            n1x1=16,

            n3x3_blue=16, n3x3=16,

            n5x5_blue=8,  n5x5=16,

            pool_planes=16

        )  # out: 16+16+16+16 = 64 (H,W 동일)



        # downsample to 8x8, widen to 128

        self.res3 = ResidualBlock(64, 128, stride=2)  # 64x16x16 -> 128x8x8

        self.res4 = ResidualBlock(128, 128, stride=1) # 128x8x8



        # head

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 10)



        ##############################################################################

        #                          END OF YOUR CODE                                  #

        ##############################################################################



    def forward(self, x):

        ##############################################################################

        #                          IMPLEMENT YOUR CODE                               #

        ##############################################################################

        x = self.stem(x)   # (B,32,32,32)

        x = self.res1(x)   # (B,64,16,16)

        x = self.res2(x)   # (B,64,16,16)

        x = self.inc(x)    # (B,64,16,16)

        x = self.res3(x)   # (B,128,8,8)

        x = self.res4(x)   # (B,128,8,8)

        x = self.gap(x)    # (B,128,1,1)

        x = torch.flatten(x, 1)  # (B,128)

        out = self.fc(x)         # (B,10)



        ##############################################################################

        #                          END OF YOUR CODE                                  #

        ##############################################################################

        return out



# initialize model

betternet = BetterNet()

betternet = betternet.to(device)



# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(betternet.parameters(), lr=0.001, momentum=0.9)



PATH = './better_net.pth'

# Train

train(betternet, dataloader_train, 10, criterion, optimizer, device, PATH)

# Test

betternet.load_state_dict(torch.load(PATH))

print_accuracy(betternet, dataloader_test)


