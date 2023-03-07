# Prints training loss, validation loss, and validation accuracy for each epoch while the network is training

# Options:
#         Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#         Choose architecture: python train.py data_dir --arch "densenet161"
#         Set hyperparameters: python train.py data_dir --learning_rate 0.001 --epochs 5 --hidden_units 2048
#         Use GPU for training: python train.py data_dir --gpu

# bash cmd for training a model: First cd to ImageClassifier folder
# python train.py './flowers' --save_directory './saved_models' --epochs 5 --model_arch densenet161 --hidden_units 2048 --gpu

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
import argparse
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json

def data_transformation(args):
    # define ImageFolder, DataLoader and Transformations
    # returns DataLoader objects for training and validation and an index dictionary for classes
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # confirms paths and directory locations
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {}".format(train_dir))
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {}".format(valid_dir))
        raise FileNotFoundError

# Transform training and validation data
    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])


    train_data = ImageFolder(root=train_dir, transform=train_transform)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transform)

    trainloader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return trainloader, validloader, train_data.class_to_idx



def train_model(args, trainloader, validloader, class_to_idx):
    # trains model, saves model to dir, prints start of training with specified device, the progress of training based on metrics, and saves model if successful.
    
    # build model using pretained densenet model    
    if args.model_arch == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
    elif args.model_arch == "densenet161":
        model = torchvision.models.densenet161(pretrained=True)

    # freeze model parameters.
    for param in model.parameters():
        param.requires_grad = False

    pretrained_model_in_features = model.classifier.in_features

    # Alter architecture depending on the desired classes to be predicted.
    classifier = nn.Sequential(nn.Linear(in_features=pretrained_model_in_features, out_features = args.hidden_units, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=0.7),
                               nn.Linear(in_features = args.hidden_units, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier
    # specify criterion.
    criterion = nn.NLLLoss()

    # specify optimizer.
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # specify the device being used for training.
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected, but no GPU is available. Training with CPU instead.")
    else:
        device = 'cpu'
    print("Using {} to train model.".format(device))
            
    # move model to device.
    model.to(device)
    
    print_every = 16
    
    steps = 0

    train_losses, validation_losses = [], []
    running_loss = 0
    
    for e in range(args.epochs):
    
    
        for ii, (train_inputs, train_labels) in enumerate(trainloader):
            steps += 1

            # Move input and label tensors to the GPU. (if available)
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        
            optimizer.zero_grad()

            # Forward and backward propagations
            outputs = model.forward(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            if steps % print_every == 0:
                
                model.eval()
                validation_loss = 0
                accuracy = 0
            
                for ii, (validation_inputs, validation_labels) in enumerate(validloader):
                    optimizer.zero_grad()
                
                    validation_inputs, validation_labels = validation_inputs.to(device), validation_labels.to(device)
                    model.to(device)
                
                    with torch.no_grad():    
                        outputs = model.forward(validation_inputs)
                        validation_loss = criterion(outputs, validation_labels)
                        ps = torch.exp(outputs).data 
                        equality = (validation_labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                validation_loss = validation_loss / len(validloader)
                train_loss = running_loss / len(trainloader)
            
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
            
                accuracy = accuracy /len(validloader)
            
                print("Epoch: {}/{}... ".format(e+1, args.epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(validation_loss),
                  "Accuracy: {:.4f}".format(accuracy))
            
                running_loss = 0
                
                
    model.class_to_idx = class_to_idx
            
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'dense_type': args.model_arch
                 }   
    
    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    return True


if __name__ == '__main__':

    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='data_directory', help="This is the directory of the training images e.g. if a sample file is in /flowers/train/.../001.png then supply /flowers. Expect different folders for, 'train' & 'valid'")

    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train with GPU using CUDA libraries", action='store_true')
    parser.add_argument('--model_arch', dest='model_arch', help="This is type of pre-trained model that will be used", default="densenet161", type=str, choices=['densenet121', 'densenet161'])
    parser.add_argument('--learning_rate', dest='learning_rate', help="Learning rate while training the model. Default is 0.001 and default data type is float", default = 0.001, type = float)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs when training the model. Default data type is int. Default number of epochs is 5.", type = int, default = 5)
    parser.add_argument('--hidden_units', dest='hidden_units', help="Specify the hidden units after model architecture", default = 2048, type = int)
    parser.add_argument('--save_directory', dest='save_directory', help="The directory where the model will be saved after training.", default='../saved_models')

    args = parser.parse_args()

    # load and transform data
    trainloader, validloader, class_to_idx = data_transformation(args)

    # train and save model
    train_model(args, trainloader, validloader, class_to_idx)