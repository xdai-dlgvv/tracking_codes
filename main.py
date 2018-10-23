import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480


# gt_center = [194, 306]
# gt_center = [131, 196]


def gaussian_2d(center):
    """ According the center to create the Gaussian 2d map
        Args:
            center: (tuple) 
        Return:
            gaussian_map: (ndarray)  the Gaussian 2d map
    """
    gaussian_map = np.zeros((init_box[3], init_box[2]))
    height, width = np.shape(gaussian_map)
    for i in range(height):
        for j in range(width):
            dis = (i - center[0]) ** 2 + (j - center[1]) ** 2
            gaussian_map[i, j] = np.exp(-dis / 200)
    return gaussian_map


def pre_process_img(img):
    """ do the pre-process to the input image
        Args:
            img: (ndarray) original image 
        Return:
            img: (ndarray) after pre-process image
    """
    # img = np.log(img)
    # img = preprocessing.minmax_scale(img, (-1, 1))
    # print()
    return img


def calculate_psr(g_, center_):
    """ calculate the PSR 
        Args:
            g_: (ndarray) the response map of the new frame
            center_: (ndarray)  the object center in the g_
        Return:
            the PSR of g_
    """
    peak = []
    sidelobe = []
    m, n = np.shape(g_)
    for i in range(m):
        for j in range(n):
            if center_[0] - 5 <= i <= center_[0] + 5 and center_[1] - 5 <= j <= center_[1] + 5:
                peak.append(g_[i, j])
            else:
                sidelobe.append(g_[i, j])
    g_max = np.max(peak)
    delta_mean = np.mean(sidelobe)
    delta_std = np.std(sidelobe)
    return (g_max - delta_mean) / delta_std


eta = 0.125
data_path = 'ball'
image_list = os.listdir(data_path)


# handle the first frame
f = cv2.imread(os.path.join(data_path, image_list[0]))
init_box = cv2.selectROI('12', f)
# print(init_box)
# init_box = [265, 136, 72, 139]
left, top = init_box[0], init_box[1]
center = init_box[0] + init_box[2] / 2, init_box[1] + init_box[3] / 2
width = init_box[2]
height = init_box[3]
crop_img = cv2.cvtColor(f[int(center[1] - height / 2):int(center[1] + height / 2),
                        int(center[0] - width / 2):int(center[0] + width / 2)], cv2.COLOR_RGB2GRAY)
crop_img = pre_process_img(crop_img)
F = np.fft.fft2(crop_img)
g = gaussian_2d((init_box[3] / 2, init_box[2] / 2,))
G1 = np.matrix(np.fft.fft2(g))
A = np.multiply(G1, F.conjugate())
B = np.multiply(F, F.conjugate())
# G = np.log(np.abs(G))


for img_path in image_list[1:]:
    f = cv2.imread(os.path.join(data_path, img_path))
    crop_img = cv2.cvtColor(f[int(center[1] - height / 2):int(center[1] + height / 2),
                            int(center[0] - width / 2):int(center[0] + width / 2)], cv2.COLOR_RGB2GRAY)

    # make sure the crop image is the same as filter H
    # this part need more operation to handle exception
    try:
        crop_img = cv2.resize(crop_img, (width, height))
    except:
        continue

    # use H to calculate the respond map
    crop_img = pre_process_img(crop_img)
    F = np.fft.fft2(crop_img)
    # F = np.log(np.abs(F))
    H = (A / B).conjugate()
    G = np.multiply(F, H.conjugate())
    g = np.fft.ifft2(G)
    g = np.array(abs(g), dtype=np.float)
    cv2.imshow('g', np.asarray(g * 255, dtype=np.uint8))
    cv2.waitKey(10)

    # according the respond map to find the max respond
    m, n = np.shape(g)
    tmp = np.max(g)
    index = int(g.argmax())
    x = int(index / n)
    y = index % n
    center = left + y, top + x
    psr = calculate_psr(g, (x, y))
    print(psr)
    left, top = int(center[0] - width / 2), int(center[1] - height / 2)
    # cv2.circle(f, (left, top), 10, (255, 0, 0))
    f = cv2.rectangle(f, (left, top), (left + width, top + height), (0, 0, 255), 2)
    cv2.imshow('12', f)
    cv2.waitKey(100)

    # update the A and B
    A = eta * np.multiply(G1, F.conjugate()) + (1 - eta) * A
    B = eta * np.multiply(F, F.conjugate()) + (1 - eta) * B
