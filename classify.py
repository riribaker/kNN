# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
from operator import itemgetter
from numpy.random import permutation
def trainPerceptron(train_set, train_labels, learning_rate, max_iter):

    # TODO: Write your code here
    W = np.zeros(len(train_set[0]))
    b  = 0

    # activation function: sign(f(x)) = 1 for true, = -1 for not true
    labels = np.zeros(len(train_set))
    for i in range(len(train_labels)):
        if train_labels[i]:
            labels[i] = 1
        else:
            labels[i] = -1
    
    for update in range(max_iter):
        # DO NOT USE PERMUTATION (dont want to shuffle data for autograder)
        for trainCount in range(len(train_set)):
            img = train_set[trainCount]
            result = np.sign(np.dot(W, img) + b)

            if result != labels[trainCount]:
                W += (learning_rate * labels[trainCount]) * img
                b += learning_rate * labels[trainCount]
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    (weights,b) = trainPerceptron(train_set,train_labels,learning_rate,max_iter)
    labels = np.zeros(len(dev_set))
    # Shuffle input data with enumerate
    for devCount, image in enumerate(dev_set):
        result = np.sign(np.dot(weights, image) + b)
        if result == 1:
            labels[devCount] = 1
    out = labels.tolist()
    return out

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    # iterate through images in input set
    devlabels = np.zeros(len(dev_set))
    for devCount in permutation(len(dev_set)):
        img = dev_set[devCount]
        neighbors = []
        for trainIdx in permutation(len(train_set)):
            #print(trainIdx)
            img_t = train_set[trainIdx]
            label_t = train_labels[trainIdx]
            #print(img_t)
            dist = np.linalg.norm(img - img_t)
            #print(dist,label_t)
            neighbors.append((dist,label_t))
        neighbors.sort(key = itemgetter(0))
        sum = 0
        for n in neighbors[0:k]:
            if n[1] == True:
                sum+=1
            else:
                sum+=-1
        if(sum>0):
            devlabels[devCount]= 1
        else:
            devlabels[devCount]=0

    out = devlabels.tolist()
    return out
