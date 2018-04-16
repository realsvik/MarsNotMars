from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os

def buildImagePaths(pathToMarsImages):
    #build mars image paths
    imgPaths = []
    i = 0
    for root, dirs, files in os.walk(pathToMarsImages):
        path = root.split(os.sep)
        for file in files:
            #print file
            imgPaths.append(pathToMarsImages+file)
    return imgPaths
	
def gatherData(imagePaths):
    # grab the image paths and randomly shuffle them
    data = []
    labels = []
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        #print imagePath
        image = cv2.imread(imagePath) 
        image = cv2.resize(image, (48, 48))
        image = img_to_array(image)
        data.append(image)
        label = 0 if "notMarsBar" in imagePath else 1
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels
	
