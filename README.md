# ImageClassifierAWSUdacityScholarship
This repository is created for demonstrating the skills obtained during the AWS AI & ML Scholarship partnered with Udacity. (https://sites.google.com/udacity.com/awsaiml-win22/home)

Jupyter notebook file (Image Classifier Project Scholarship Submission.ipynb) contains the python code for training a computer vision deep learning model that uses a pretrained densenet161 architecture.

The SOTA version shows the steps for the development of a SOTA (State of the Art) model with Swin Transformer architecture that is trained with 224 x 224 pixel images. Current state of the art is trained with 384 x 384 pixel images.

The train and predict files are command line application tools that can be run in local clusters to train and predict images locally, although they won't be able to replicate the SOTA results, since they are designed to work on earlier versions of python and pytorch environments.

cat_to_name.json file contains the dictionary for matching class names to index names for the 102 different species used for training the model.

The SOTA model uses the following image augmentations:

1-) Randomly crop and resize (to 224 by 224 pixels), flip and rotate images, 

2-) Adjust the color, brightness, saturation, hue 

3-) Apply gamma correction. 

4-) Create and normalize a tensor

5-)Finally, erase a portion of the image depending on the specified range.

<img width="596" alt="image" src="https://user-images.githubusercontent.com/16454824/223679385-9334c0fa-37fc-4a3d-b7b2-27f3d019a175.png">

The model is 99.76% accurate and uses SwinTransformer architecture, Focal loss, learning rate schedulers and cutmix-fmix for achieving these results. In the final version, I decided to remove cutmix and use random erasing augmentation instead, since it gave more stable results.

<img width="544" alt="image" src="https://user-images.githubusercontent.com/16454824/223685636-d2d71e68-6f93-4e30-84f7-357f650c4204.png">

![image](https://user-images.githubusercontent.com/16454824/223685686-eba2b73c-1cbd-4917-b30d-5aaf6dc15462.png)


The model only predicts two images wrong in the test dataset, which consists of 832 images.

Those images are:

<img width="287" alt="image" src="https://user-images.githubusercontent.com/16454824/223683779-427fd9f0-13b7-4e27-abb7-c592ea1880e2.png">

The model predicts these images as mallow but they belong to camellia species.

You can check the state of the art models for the same dataset from the following site: https://paperswithcode.com/sota/image-classification-on-flowers-102




