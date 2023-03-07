#     Options and usage:
#         Return most likely classes in orderof descending probabilities: python predict.py input image checkpoint file location --top_k 10 
#         Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#         Use GPU for inference: python predict.py input checkpoint --gpu

# bash cmd for experiment: python predict.py './flowers/test/19/image_06159.jpg' './saved_models/checkpoint.pth' --top_k 10 --gpu


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


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['dense_type'] == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
    elif checkpoint['dense_type'] == "densenet161":
        model = torchvision.models.densenet161(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
   # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image_path) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    img = np.array(transform(img))

    return img


def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)

    # move to device
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    ps, top_classes = ps.topk(topk, dim=1)
    
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    # returning both as lists instead of torch objects for simplicity
    return ps.tolist()[0], predicted_flowers_list

def print_predictions(args):
    # load model
    model = load_checkpoint(args.model_filepath)

    # decide device depending on user arguments and device availability
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("No GPU is available. Using CPU instead.")
    else:
        device = 'cpu'

    model = model.to(device)

    # print(model.class_to_index)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # predict image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
          print("#{: <3} {: <25} Prob: {:.2f}%".format(i+1, top_classes[i], top_ps[i]*100))
    
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="This is a image file to classify")
    parser.add_argument(dest='model_filepath', help="This is file path of a checkpoint file obtained after training")

    # optional arguments
    parser.add_argument('--top_k', dest='top_k', help="Most likely classes to return with probabilities, default is 5",type=int, default=5)
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="This is a file path to a json file that maps categories to class names", default='cat_to_name.json')
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to predict using GPU via CUDA", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)
