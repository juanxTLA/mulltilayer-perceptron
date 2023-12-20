import wget
import gzip
import shutil
import numpy as np
import os

'''
    This file is used to get the dataSet files. They are downloaded from the internet with wget
    and are then stored in the working direct
'''


def dowloadMNISTData():
    url = 'http://yann.lecun.com/exdb/mnist/'
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for filename in filenames:
        wget.download(url + filename)


    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for filename in filenames:
        with gzip.open(filename, 'rb') as f_in:
            with open(filename[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def getImages(type):
    #returns wither train or test images
    cwd = os.getcwd()
    if(type == 'train'):
        path = os.path.join(cwd, 'train', 'images', 'train-images-idx3-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28)

    else:
        path = os.path.join(cwd, 'test', 'images', 't10k-images-idx3-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28)
    
    images = np.reshape(images, (images.shape[0], -1)) #vectorize
    images = images / 255.0

    return images

def getLabels(type):
    #returns either test or train labels as one-hot encoded collection of vectors
    cwd = os.getcwd()
    if(type == 'train'):
        path = os.path.join(cwd, 'train', 'labels', 'train-labels-idx1-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    else:
        path = os.path.join(cwd, 'test', 'labels', 't10k-labels-idx1-ubyte.gz')
        with gzip.open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    num_classes = 10
    num_samples = len(labels)

    one_hot_labels = np.zeros((num_samples, num_classes))
    one_hot_labels[np.arange(num_samples), labels] = 1
    
    return one_hot_labels




