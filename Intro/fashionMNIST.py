import torch
import torch.nn as nn
import torchvision                                                       
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
import os.path

from saving import generate_unique_logpath, ModelCheckpoint
from DatasetTransformer import DatasetTransformer
from Linear import LinearNet, FullyConnected, FullyConnectedRegularized
from Convolutional import SimpleConv
from training_tools import train, test, compute_mean_std

from torch.utils.tensorboard import SummaryWriter    

num_threads = 12     # Loading the dataset is using 12 CPU threads
batch_size = 128   # Using minibatches of 128 samples
nsamples = 10
classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
epochs = 5

top_logdir = "./logs"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir = generate_unique_logpath(top_logdir, "simple_convolutional")
print("Logging to {}".format(logdir))

if not os.path.exists(logdir):
    os.mkdir(logdir)

## Datasets

dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

# Load the dataset for the training/validation sets
train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                           train=True,
                                           transform= None, #transforms.ToTensor(),
                                           download=True)

# Split it into training and validation sets
nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
nb_valid =  int(valid_ratio * len(train_valid_dataset))
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

# Load the test set 
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                 transform= None,
                                                 train=False)

## Dataset normalization
normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
normalizing_loader = torch.utils.data.DataLoader(dataset=normalizing_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_threads)

# Compute mean and variance from the training set
mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)                                       

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean_train_tensor)/std_train_tensor)
])

train_dataset = DatasetTransformer(train_dataset, data_transforms)
valid_dataset = DatasetTransformer(valid_dataset, data_transforms)
test_dataset  = DatasetTransformer(test_dataset , data_transforms)

## Dataloaders

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,                # <-- this reshuffles the data at every epoch
                                          num_workers=num_threads)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=num_threads)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_threads)


print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

imgs, labels = next(iter(train_loader))

#fig=plt.figure(figsize=(20,5),facecolor='w')
#for i in range(nsamples):
#    ax = plt.subplot(1,nsamples, i+1)
#    plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
#    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)

#plt.savefig('fashionMNIST_samples.png', bbox_inches='tight')
#plt.show()

model = SimpleConv(1*28*28, 10)
f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

tensorboard_writer   = SummaryWriter(log_dir = logdir)

use_gpu = torch.cuda.is_available()

if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)
model_checkpoint = ModelCheckpoint(logdir + "/best_model.pt", model)

for t in range(epochs):
    print("Epoch {}".format(t))
    sys.stdout.write('Training progression ')
    train_loss, train_acc = train(model, train_loader, f_loss, optimizer, device)
    sys.stdout.write('Testing progression ')
    val_loss, val_acc = test(model, valid_loader, f_loss, device)
    sys.stdout.write('\n')
    
    model_checkpoint.update(val_loss)

    tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
    tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
    tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
    tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)

## Load the best model

model_path = logdir + "/best_model.pt"
model = SimpleConv(1*28*28)

model = model.to(device)

model.load_state_dict(torch.load(model_path))

# Switch to eval mode 
model.eval()

test_loss, test_acc = test(model, test_loader, f_loss, device)
print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))
