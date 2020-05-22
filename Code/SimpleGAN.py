
# To implement a GAN, we basically require 5 components:

    # Real Dataset (real distribution)
    # Low dimensional random noise that is input to the Generator to produce fake images
    # Generator that generates fake images
    # Discriminator that acts as an expert to distinguish real and fake images.
    # Training loop where the competition occurs and models better themselves.

import torch
import random
import numpy as np
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

## Checks for the availability of GPU
if torch.cuda.is_available():
    print("working on gpu!")
    device = 'cuda'
else:
    print("No gpu! only cpu ;)")
    device = 'cpu'

if device == 'cpu':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
elif device == 'cuda':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '0'

# Data Preparation
    # Normalize images to range [-1,1]
if not os.path.isdir('./data'):
    os.mkdir('./data')
root = './data/'

train_bs = 128

transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize(mean=[0.5],
                                std=[0.5])
        ])

training_data = torchvision.datasets.MNIST(root, train=True, transform=transform,download=True)
train_loader=torch.utils.data.DataLoader(dataset=training_data, batch_size=train_bs, shuffle=True, drop_last=True)

# Let us define a function which takes (batchsize, dimension) as input and returns a random noise of requested dimensions. \
# This noise tensor will be the input to the generator.
def noise(bs, dim):
    """Generate random Gaussian noise.

    Inputs:
    - bs: integer giving the batch size of noise to generate.
    - dim: integer giving the dimension of the the noise to generate.

    Returns:
    A PyTorch Tensor containing Gaussian noise with shape [bs, dim]
    """

    out = (torch.randn((bs, dim))).to(device)
    return out


# Generator architecture:
    # noise_dim -> 256
    # LeakyReLU (works well for the Generators)
    # 256 -> 512
    # LeakyReLU
    # 512 -> 1024
    # LeakyReLU
    # 1024 -> out_size(784)
    # TanH
    # LeakyRELU: https://pytorch.org/docs/stable/nn.html#leakyrelu
    # Fully connected layer: https://pytorch.org/docs/stable/nn.html#linear
    # TanH activation: https://pytorch.org/docs/stable/nn.html#tanh

class Generator(nn.Module):
    def __init__(self, noise_dim=100, out_size=784):
        super(Generator, self).__init__()

        self.layer1 = nn.Linear(noise_dim, 256)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, out_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''

        Make a forward pass of the input through the generator. Leaky relu is used as the activation
        function in all the intermediate layers. Tanh activation function is only used at the end (which
        means only after self.layer4)

        Note that, generator takes an random noise as input and gives out fake "images". Hence, the output
        after tanh activation function is reshaped into the same size as the real images. i.e.,
        [batch_size, n_channels, H, W] == (batch_size, 1,28,28)

        '''
        x = self.layer1(x)
        x = self.leakyrelu(x)
        x = self.layer2(x)
        x = self.leakyrelu(x)
        x = self.layer3(x)
        x = self.leakyrelu(x)
        x = self.layer4(x)
        x = self.tanh(x)
        #         x = x.view(train_bs, 1, 28, 28)

        return x

generator = Generator().to(device)

# Discriminator architecture:

    # input_size->512
    # LeakyReLU with negative slope = 0.2
    # 512 -> 256
    # LeakyReLU with negative slope = 0.2
    # 256->1

## Similar to the Generator, we now define a Discriminator which takes in a vector and output a single scalar
## value.

class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Linear(input_size, 512)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):

        '''

        The Discriminator takes a vectorized input of the real and generated fake images. Reshape the input
        to match the Discriminator architecture.

        Make a forward pass of the input through the Discriminator and return the scalar output of the
        Discriminator.

        '''

        y = self.layer1(x)
        y = self.leakyrelu(y)
        y = self.layer2(y)
        y = self.leakyrelu(y)
        y = self.layer3(y)

        return y

discriminator = Discriminator()
discriminator = discriminator.to(device)

# Loss Fuction
# Binary cross entropy loss function. The loss function includes sigmoid activation followed by logistic loss.
# Binary cross entropy loss with logits: https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
bce_loss = nn.BCEWithLogitsLoss()

def DLoss(logits_real, logits_fake, targets_real, targets_fake):
    '''
    d1 - binary cross entropy loss between outputs of the Discriminator with real images
         (logits_real) and targets_real.

    d2 - binary cross entropy loss between outputs of the Discriminator with the generated fake images
         (logits_fake) and targets_fake.

    '''
    d1 = bce_loss(logits_real, targets_real)
    d2 = bce_loss(logits_fake, targets_fake)
    total_loss = d1 + d2
    return total_loss

logits_real = torch.ones([10, 64], dtype=torch.float32)
logits_fake = torch.full([10, 64], 0.999)
targets_real = torch.ones([10, 64], dtype=torch.float32)
targets_fake = torch.full([10, 64], 0.999)

DLoss(logits_real, logits_fake, targets_real, targets_fake)


def GLoss(logits_fake, targets_real):
    '''
    The aim of the Generator is to fool the Discriminator into "thinking" the generated images are real.

    g_loss - binary cross entropy loss between the outputs of the Discriminator with the generated fake images
         (logits_fake) and targets_real.

    Thus, the gradients estimated with the above loss corresponds to generator producing fake images that
    fool the discriminator.

    '''
    g_loss = bce_loss(logits_fake, targets_real)
    return g_loss

# Optimizer
    # Optimizers for training the Generator and the Discriminator.
    # Adam optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
noise_dim = 100

## Training loop
discriminator_loss = 0.0
generator_loss = 0.0
for epoch in range(epochs):
    print("Epoch :", epoch + 1)
    for i, (images, _) in enumerate(train_loader):
        # We set targets_real and targets_fake to non-binary values(soft and noisy labels).
        # This is a hack for stable training of GAN's.
        # GAN hacks: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels

        targets_real = (torch.FloatTensor(images.size(0), 1).uniform_(0.8, 1.0)).to(device)
        targets_fake = (torch.FloatTensor(images.size(0), 1).uniform_(0.0, 0.2)).to(device)

        images = images.to(device)

        # YOUR CODE HERE
        #         raise NotImplementedError()

        ## D-STEP:
        ## First, clear the gradients of the Discriminator optimizer.

        ## Estimate logits_real by passing images through the Discriminator

        ## Generate fake_images by passing random noise through the Generator. Also, .detach() the fake images
        ## as we don't compute the gradients of the Generator when optimizing Discriminator.
        ## fake_images = generator(noise(train_bs, noise_dim)).detach()

        ## Estimate logits_fake by passing the fake images through the Discriminator

        ## Compute the Discriminator loss by calling DLoss function.

        ## Compute the gradients by backpropagating through the computational graph.

        ## Update the Discriminator parameters.

        optimizer_D.zero_grad()
        images = images.view(images.size(0), 784)
        logits_real = discriminator(images)
        fake_images = generator(noise(train_bs, noise_dim)).detach()
        logits_fake = discriminator(fake_images)
        d_loss = DLoss(logits_real, logits_fake, targets_real, targets_fake)
        d_loss.backward()
        optimizer_D.step()
        discriminator_loss = discriminator_loss + d_loss.item()

        ## G-STEP:
        ## clear the gradients of the Generator.

        ## Generate fake images by passing random noise through the Generator.

        ## Estimate logits_fake by passing the fake images through the Discriminator.

        ## compute the Generator loss by caling GLoss.

        ## compute the gradients by backpropagating through the computational graph.

        ## Update the Generator parameters.

        optimizer_G.zero_grad()
        fake_images = generator(noise(train_bs, noise_dim))
        logits_fake = discriminator(fake_images)
        g_loss = GLoss(logits_fake, targets_real)
        g_loss.backward()
        optimizer_G.step()
        generator_loss = generator_loss + g_loss.item()
        #         print("shape",fake_images.shape)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    #         print("shape",fake_images.shape)

    #     print("D Loss: ", discriminator_loss.item())
    #     print("G Loss: ", generator_loss.item())
    print("D Loss: ", d_loss.item())
    print("G Loss: ", g_loss.item())

    if epoch % 2 == 0:
        viz_batch = fake_images.data.cpu().numpy()
        fig = plt.figure(figsize=(8, 10))
        for i in np.arange(1, 10):
            ax = fig.add_subplot(3, 3, i)
            img = viz_batch[i].squeeze()
            plt.imshow(img)
        plt.show()