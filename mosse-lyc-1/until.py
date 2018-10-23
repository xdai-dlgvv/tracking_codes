import numpy as np
import cv2

def preprocessing(image):
    height , weight =image.shape
    image = np.log(image+1)
    image = (image - np.mean(image)) / (np.std(image) + 1e-5)
    image = coswindows(image, height, weight)
    return image

def coswindows(image, height, weight):
    w = np.hanning(weight)
    h = np.hanning(height)
    ww, hh = np.meshgrid(w,h)
    image = image *(ww*hh)
    return image

def gauss(image,gt,sigma) :
    height, weight = image.shape
    w = np.arange(weight)
    h = np.arange(height)
    xx, yy = np.meshgrid(w,h)
    x0 = gt[0] + gt[2]/2
    y0 = gt[1] + gt[3]/2
    gauss_image = np.exp(-((np.square(xx-x0) + np.square(yy-y0))/(2*np.square(sigma))))
    gauss_image = normal(gauss_image)
    return gauss_image

def normal(image) :
    max_value = image.max()
    min_value = image.min()

    a = 1 / (max_value - min_value)
    b = 1 - max_value*a
    return a*image+b

def random_xuanzhuang(image) :
    #b = image.shape
    a = np.random.uniform(-30,30)
    #a = -180 / 16
    #b = 180 / 16
    #r = a + (b - a) * np.random.uniform()
    M = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),a,1)
    image = cv2.warpAffine(np.uint8(image*255), M, (image.shape[1],image.shape[0]))
    return (image.astype(np.float32)/255)
